"""Generate a ScenA reference-conditioned audio scene.

Downloads the checkpoint + audio VAE from Hugging Face (mifinkelson/scena) on first
run, then generates a short two-speaker dialogue example. Point --gemma-root at a
local copy of google/gemma-3-12b-it.

    python examples/generate.py --gemma-root /path/to/gemma-3-12b-it
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from ltx_pipelines.t2aud_ref_cond import T2AudRefCondPipeline

DIALOGUE = (
    'The speaker from reference 1 says: "The taxi drivers are on strike again." '
    'The speaker from reference 2 says: "What for?" '
    'The speaker from reference 1 says: "They want the government to reduce the price of the gasoline." '
    'The speaker from reference 2 says: "It is really a hot potato."'
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a ScenA audio scene.")
    ap.add_argument("--gemma-root", required=True, help="local path to google/gemma-3-12b-it")
    ap.add_argument("--ckpt-dir", default=None, help="dir with scena.safetensors + audio_vae.safetensors "
                    "(downloaded from HF if omitted)")
    ap.add_argument("--hf-repo", default="mifinkelson/scena")
    ap.add_argument("--prompt", default=DIALOGUE)
    ap.add_argument("--duration", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    here = Path(__file__).resolve().parent

    ckpt_dir = args.ckpt_dir or snapshot_download(
        args.hf_repo, allow_patterns=["scena.safetensors", "audio_vae.safetensors"]
    )
    ckpt_dir = Path(ckpt_dir)

    pipe = T2AudRefCondPipeline(
        checkpoint_path=str(ckpt_dir / "scena.safetensors"),
        audio_vae_path=str(ckpt_dir / "audio_vae.safetensors"),
        gemma_root=args.gemma_root,
    )
    audio = pipe(
        prompt=args.prompt,
        ref_audio_paths=[
            str(here / "references" / "reference_1.wav"),
            str(here / "references" / "reference_2.wav"),
        ],
        duration=args.duration,
        seed=args.seed,
    )
    audio.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
