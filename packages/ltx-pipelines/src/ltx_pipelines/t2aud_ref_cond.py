"""Reference-conditioned text-to-audio pipeline (ScenA).

Generate a multi-speaker audio scene from a text prompt plus one or more
reference-audio clips that set the speakers' voices. Built on the LTX-2
audio-only transformer with reference conditioning.

Example::

    from ltx_pipelines.t2aud_ref_cond import T2AudRefCondPipeline

    pipe = T2AudRefCondPipeline(
        checkpoint_path="checkpoints/scena.safetensors",
        audio_vae_path="checkpoints/audio_vae.safetensors",
        gemma_root="/path/to/google/gemma-3-12b-it",
    )
    audio = pipe(
        prompt='The speaker from reference 1 says "hello" then reference 2 replies "hi there".',
        ref_audio_paths=["examples/references/reference_1.wav", "examples/references/reference_2.wav"],
        duration=8.0,
    )
    audio.save("out.wav")
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae.audio_vae import AudioEncoder, encode_audio as vae_encode_audio
from ltx_core.model.transformer import LTXV_AUDIO_ONLY_RENAMING_MAP, LTXAudioOnlyModelConfigurator
from ltx_core.pipeline.ref_conds import EncodedRefCond
from ltx_core.text_encoders.gemma import (
    SCENA_AUDIO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS,
    ScenaAudioOnlyEmbeddingsProcessorConfigurator,
)
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape
from ltx_pipelines.utils import get_device
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, DiffusionStage, PromptEncoder
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, RefCondDenoiser
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.types import ModalitySpec

logger = logging.getLogger(__name__)

# Audio VAE latent frame rate: sample_rate / hop_length / audio_latent_downsample_factor.
_AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4  # = 25.0
_PLACEHOLDER_RES = 512  # video dims are unused in audio-only generation


def _encode_ref_audio(
    audio: Audio, encoder: AudioEncoder, device: torch.device, dtype: torch.dtype
) -> EncodedRefCond:
    """Encode one reference waveform through the audio VAE and patchify to tokens."""
    # The audio VAE is stereo; duplicate mono references to two channels.
    wf = audio.waveform
    if wf.ndim == 3 and wf.shape[1] == 1:
        audio = replace(audio, waveform=wf.repeat(1, 2, 1))
    elif wf.ndim == 2 and wf.shape[0] == 1:
        audio = replace(audio, waveform=wf.repeat(2, 1))
    latent = vae_encode_audio(audio, encoder, None).to(dtype=dtype, device=device)  # (B, C, T, F)
    b, c, t, f = latent.shape
    tools = AudioLatentTools(AudioPatchifier(patch_size=1), AudioLatentShape(batch=b, channels=c, frames=t, mel_bins=f))
    state = tools.create_initial_state(device=device, dtype=dtype, initial_latent=latent)
    return EncodedRefCond(latents=state.latent, positions=state.positions)


class T2AudRefCondPipeline:
    """Text-to-audio generation conditioned on reference speaker voices (ScenA)."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        audio_vae_path: str | None = None,
        max_ref_conds: int = 3,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self.max_ref_conds = max_ref_conds
        self._scheduler = LTX2Scheduler()
        # The audio VAE + vocoder live in the bundled audio_vae.safetensors; fall back to
        # the main checkpoint if a combined file is provided.
        audio_vae_path = audio_vae_path or checkpoint_path

        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            embeddings_processor_configurator=ScenaAudioOnlyEmbeddingsProcessorConfigurator,
            embeddings_processor_sd_ops=SCENA_AUDIO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras or []),
            model_configurator=LTXAudioOnlyModelConfigurator,
            model_sd_ops=LTXV_AUDIO_ONLY_RENAMING_MAP,
        )
        self.audio_conditioner = AudioConditioner(checkpoint_path=audio_vae_path, dtype=self.dtype, device=self.device)
        self.audio_decoder = AudioDecoder(checkpoint_path=audio_vae_path, dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        ref_audio_paths: list[str | Path],
        negative_prompt: str = "",
        seed: int = 42,
        duration: float = 8.0,
        num_inference_steps: int = 60,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory | None = None,
    ) -> Audio:
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(cfg_scale=7.0)
        if len(ref_audio_paths) > self.max_ref_conds:
            raise ValueError(f"Got {len(ref_audio_paths)} references but max_ref_conds={self.max_ref_conds}")

        # 1. Encode reference audios through the audio VAE.
        logger.info("Encoding %d reference audio(s)...", len(ref_audio_paths))
        ref_audios = []
        for p in ref_audio_paths:
            decoded = decode_audio_from_file(str(p), self.device)
            if decoded is None:
                raise ValueError(f"Failed to decode reference audio: {p}")
            ref_audios.append(decoded)

        def _encode_refs(encoder: AudioEncoder) -> dict[int, EncodedRefCond]:
            encoded = {}
            for idx, a in enumerate(ref_audios):
                encoded[idx] = _encode_ref_audio(a, encoder, self.device, self.dtype)
                logger.info("  reference %d: %d tokens", idx + 1, encoded[idx].latents.shape[1])
            return encoded

        encoded_refs = self.audio_conditioner(_encode_refs)

        # 2. Encode the text. ScenA's Gemma readout lands in the video_connector slot,
        #    so the audio cross-attention context is ``video_encoding``. An empty negative
        #    prompt yields the unconditional (null) embedding used for CFG.
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        a_context_p = ctx_p.video_encoding
        a_context_n = ctx_n.video_encoding

        # 3. Reference-conditioned CFG denoiser.
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params, negative_context=a_context_n
        )
        denoiser = RefCondDenoiser(
            inner_denoiser=FactoryGuidedDenoiser(
                v_context=None,
                a_context=a_context_p,
                video_guider_factory=None,
                audio_guider_factory=audio_guider_factory,
            ),
            ref_conds=encoded_refs,
            max_ref_conds=self.max_ref_conds,
        )

        # 4. Sample. ``frames``/``fps`` encode the target duration (video dims unused);
        #    audio latent length = round((frames / fps) * 25).
        sigmas = self._scheduler.execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        num_frames = round(duration * _AUDIO_LATENTS_PER_SECOND)
        _, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(seed)),
            width=_PLACEHOLDER_RES,
            height=_PLACEHOLDER_RES,
            frames=num_frames,
            fps=_AUDIO_LATENTS_PER_SECOND,
            video=None,
            audio=ModalitySpec(context=a_context_p),
        )

        # 5. Decode latents to a waveform.
        return self.audio_decoder(audio_state.latent)
