# LTX-2 Pipelines

High-level pipeline implementations for generating audio-video content with Lightricks' **LTX-2** model. This package provides ready-to-use pipelines for text-to-video, image-to-video, video-to-video, and keyframe interpolation tasks.

Pipelines are built using building blocks from [`ltx-core`](../ltx-core/) (schedulers, guiders, noisers, patchifiers) and handle the complete inference flow including model loading, encoding, decoding, and file I/O.

---

## 📋 Overview

LTX-2 Pipelines provides production-ready implementations that abstract away the complexity of the diffusion process, model loading, and memory management. Each pipeline is optimized for specific use cases and offers different trade-offs between speed, quality, and memory usage.

**Key Features:**

- 🎬 **Multiple Pipeline Types**: Text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, and retake
- ⚡ **Optimized Performance**: Support for FP8 transformers, gradient estimation, and memory optimization
- 🎯 **Production Ready**: Two-stage pipelines for best quality output
- 🔧 **LoRA Support**: Easy integration with trained LoRA adapters
- 📦 **Self-Contained**: Handles model loading, encoding, decoding, and file I/O
- 🚀 **CLI Support**: All pipelines can be run as command-line scripts

---

## 🚀 Quick Start

`ltx-pipelines` provides ready-made inference pipelines for text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, and retake. Built using building blocks from [`ltx-core`](../ltx-core/), these pipelines handle the complete inference flow including model loading, encoding, decoding, and file I/O.

## 🔧 Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-pipelines
```

### Running Pipelines

All pipelines can be run directly from the command line. Each pipeline module is executable:

```bash
# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 0.8 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# View all available options for any pipeline
python -m ltx_pipelines.ti2vid_two_stages --help
```

Available pipeline modules:

- `ltx_pipelines.ti2vid_two_stages` - Two-stage text/image-to-video (recommended).
- `ltx_pipelines.ti2vid_two_stages_hq` - Two-stage text/image-to-video (different sampler, better quality).
- `ltx_pipelines.ti2vid_one_stage` - Single-stage text/image-to-video.
- `ltx_pipelines.t2a_one_stage` - Single-stage text-to-audio (audio-only output).
- `ltx_pipelines.distilled` - Fast text/image-to-video pipeline using only the distilled model.
- `ltx_pipelines.ic_lora` - Video-to-video with IC-LoRA.
- `ltx_pipelines.keyframe_interpolation` - Keyframe interpolation.
- `ltx_pipelines.a2vid_two_stage` - Audio-to-video generation conditioned on an input audio.
- `ltx_pipelines.retake` - Regenerate a time region of an existing video.
- `ltx_pipelines.hdr_ic_lora` - Video-to-video with HDR output (linear float via LogC3 inverse decode).
- `ltx_pipelines.lipdub` - Lip dubbing / re-voicing with IC-LoRA and audio reference conditioning.

Use `--help` with any pipeline module to see all available options and parameters.

---

## 🎯 Pipeline Selection Guide

### Quick Decision Tree

```text
Do you have an existing video to modify?
├─ YES → Use RetakePipeline (regenerate a specific time region)
│
Do you have an audio file to drive generation?
├─ YES → Use A2VidPipelineTwoStage (audio-to-video)
│
Do you need HDR output (linear float frames for EXR / tonemapping)?
├─ YES → Use HDRICLoraPipeline (video-to-video with LogC3 inverse decode)
│
Do you need to condition on existing images/videos?
├─ YES → Do you have reference videos for video-to-video?
│  ├─ YES → Use ICLoraPipeline
│  └─ NO → Do you have multiple keyframe images to interpolate?
│     ├─ YES → Use KeyframeInterpolationPipeline
│     └─ NO → Use TI2VidTwoStagesPipeline (image conditioning only)
│
└─ NO → Text-to-video only
   ├─ Do you need best quality?
   │  └─ YES → Use TI2VidTwoStagesPipeline (recommended for production)
   │
   └─ Do you need fastest inference?
      └─ YES → Use DistilledPipeline (with 8 predefined sigmas)
```

> **Note:** [`TI2VidOneStagePipeline`](src/ltx_pipelines/ti2vid_one_stage.py) is primarily for educational purposes. For best quality, use two-stage pipelines ([`TI2VidTwoStagesPipeline`](src/ltx_pipelines/ti2vid_two_stages.py), [`TI2VidTwoStagesHQPipeline`](src/ltx_pipelines/ti2vid_two_stages_hq.py), [`ICLoraPipeline`](src/ltx_pipelines/ic_lora.py), [`KeyframeInterpolationPipeline`](src/ltx_pipelines/keyframe_interpolation.py), [`A2VidPipelineTwoStage`](src/ltx_pipelines/a2vid_two_stage.py), or [`DistilledPipeline`](src/ltx_pipelines/distilled.py)). For editing existing videos, use [`RetakePipeline`](src/ltx_pipelines/retake.py).

### Features Comparison

| Pipeline | Stages | [Multimodal Guidance](#%EF%B8%8F-multimodal-guidance) | Upsampling | Conditioning | Best For |
| -------- | ------ | --- | ---------- | ------------- | -------- |
| **TI2VidTwoStagesPipeline** | 2 | ✅ | ✅ | Image | **Production quality** (recommended) |
| **TI2VidTwoStagesHQPipeline** | 2 | ✅ | ✅ | Image | Same as above, res_2s sampler (higher quality) |
| **TI2VidOneStagePipeline** | 1 | ✅ | ❌ | Image | Educational, prototyping |
| **DistilledPipeline** | 2 | ❌ | ✅ | Image | Fastest inference (8 sigmas) |
| **ICLoraPipeline** | 2 | ✅ | ✅ | Image + Video | Video-to-video transformations |
| **KeyframeInterpolationPipeline** | 2 | ✅ | ✅ | Keyframes | Animation, interpolation |
| **A2VidPipelineTwoStage** | 2 | ✅ | ✅ | Audio + Image | Audio-driven video generation |
| **RetakePipeline** | 1 | ✅ | ❌ | Source Video | Regenerating a time region of a video |
| **HDRICLoraPipeline** | 2 | ❌ | ✅ | Video | HDR video-to-video (linear float output for EXR) |
| **LipDubPipeline** | 2 | ✅ | ✅ | Video + Audio | Lip dubbing with audio ref conditioning |

---

## 📦 Available Pipelines

### 1. TI2VidTwoStagesPipeline

**Best for:** High-quality text/image-to-video generation with upsampling. **Recommended for production use.**

**Source**: [`src/ltx_pipelines/ti2vid_two_stages.py`](src/ltx_pipelines/ti2vid_two_stages.py)

Two-stage generation: Stage 1 generates low-resolution video with [multimodal guidance](#%EF%B8%8F-multimodal-guidance), Stage 2 upsamples to 2x resolution with distilled LoRA refinement. Supports image conditioning. Highest quality output, slower than one-stage but significantly better quality.

**Use when:** Production-quality video generation, higher resolution needed, quality over speed, text-to-video with image conditioning.

---

### 2. TI2VidTwoStagesHQPipeline

**Best for:** Same two-stage text/image-to-video as TI2VidTwoStagesPipeline but with a different sampler and step count.

**Source**: [`src/ltx_pipelines/ti2vid_two_stages_hq.py`](src/ltx_pipelines/ti2vid_two_stages_hq.py)

Uses the **res_2s** second-order sampler instead of Euler. Same stage structure (stage 1 at target resolution with CFG, stage 2 upsampling with distilled LoRA) and image conditioning support. Typically allows fewer steps for comparable quality; trade-offs differ from the default Euler-based pipeline.

**Use when:** You want the same two-stage workflow with fewer steps or prefer the res_2s sampling behavior.

---

### 3. TI2VidOneStagePipeline

**Best for:** Educational purposes and quick prototyping.

**Source**: [`src/ltx_pipelines/ti2vid_one_stage.py`](src/ltx_pipelines/ti2vid_one_stage.py)

> **⚠️ Important:** This pipeline is primarily for educational purposes. For production-quality results, use `TI2VidTwoStagesPipeline` or other two-stage pipelines.

Single-stage generation (no upsampling) with [multimodal guidance](#%EF%B8%8F-multimodal-guidance) and image conditioning support. Faster inference but lower resolution output (typically 512x768).

**Use when:** Learning how the pipeline works, quick prototyping, testing, or when high resolution is not needed.

---

### 4. DistilledPipeline

**Best for:** Fastest inference with good quality using a distilled model with predefined sigma schedule.

**Source**: [`src/ltx_pipelines/distilled.py`](src/ltx_pipelines/distilled.py)

Two-stage generation with 8 predefined sigmas (8 steps in stage 1, 4 steps in stage 2). No guidance required. Fastest inference among all pipelines. Supports image conditioning. Requires spatial upsampler.

**Use when:** Fastest inference is critical, batch processing many videos, or when you have a distilled model checkpoint.

---

### 5. ICLoraPipeline

**Best for:** Video-to-video and image-to-video transformations using IC-LoRA.

**Source**: [`src/ltx_pipelines/ic_lora.py`](src/ltx_pipelines/ic_lora.py)

Two-stage generation with IC-LoRA support. Can condition on reference videos (video-to-video) or images at specific frames. CFG guidance in stage 1, upsampling in stage 2. Requires IC-LoRA trained model.

**Note:** ICLoraPipeline can only be used with a distilled model.

**Use when:** Video-to-video transformations, image-to-video with strong control, or when you have reference videos to guide generation.

---

### 6. KeyframeInterpolationPipeline

**Best for:** Generating videos by interpolating between keyframe images.

**Source**: [`src/ltx_pipelines/keyframe_interpolation.py`](src/ltx_pipelines/keyframe_interpolation.py)

Two-stage generation with keyframe interpolation. Uses guiding latents (additive conditioning) instead of replacing latents for smoother transitions. [Multimodal guidance](#%EF%B8%8F-multimodal-guidance) in stage 1, upsampling in stage 2.

**Use when:** You have keyframe images and want to interpolate between them, creating smooth transitions, or animation/motion interpolation tasks.

---

### 7. A2VidPipelineTwoStage

**Best for:** Generating video driven by an input audio.

**Source**: [`src/ltx_pipelines/a2vid_two_stage.py`](src/ltx_pipelines/a2vid_two_stage.py)

Two-stage audio-to-video generation. Stage 1 generates video at half resolution with audio conditioning (video-only denoising with the audio frozen), then Stage 2 upsamples by 2x and refines the video while keeping the audio fixed, using a distilled LoRA. The input audio is encoded via the audio VAE and used as the initial audio latent, but the original audio waveform is passed through and returned in the output to preserve fidelity. Supports image conditioning and prompt enhancement.

**Extra CLI arguments:** `--audio-path` (required), `--audio-start-time`, `--audio-max-duration`.

**Use when:** You have an audio clip and want to generate a matching video, audio-reactive video generation, or music visualization.

---

### 8. RetakePipeline

**Best for:** Regenerating a specific time region of an existing video while keeping the rest unchanged.

**Source**: [`src/ltx_pipelines/retake.py`](src/ltx_pipelines/retake.py)

Single-stage generation that encodes the source video and audio into latents, applies a temporal region mask to mark `[start_time, end_time]` for regeneration, and denoises only the masked region from a text prompt. Content outside the time window is preserved. Supports independent control over video and audio regeneration (`regenerate_video`, `regenerate_audio` flags), and can use either the full model with CFG guidance or the distilled model with a fixed sigma schedule.

**Extra CLI arguments:** `--video-path` (required), `--start-time` (required), `--end-time` (required).

**Constraints:** Source video frame count must satisfy the 8k+1 format (e.g. 97, 193) and resolution must be multiples of 32.

**Use when:** You want to re-do a specific section of a generated video (e.g. fix a bad segment), selectively regenerate audio or video in a time window, or iterate on part of a result without re-generating the entire clip.

---

### 9. HDRICLoraPipeline

**Best for:** Video-to-video generation with HDR output for EXR export and offline tonemapping.

**Source**: [`src/ltx_pipelines/hdr_ic_lora.py`](src/ltx_pipelines/hdr_ic_lora.py)

Two-stage video-to-video on the distilled model with an HDR IC-LoRA. Decoded latents pass through an HDR inverse transform (ARRI LogC3, auto-detected from LoRA metadata) to produce a **linear HDR float** tensor `[f, h, w, c]`. Video-only (audio skipped). Text embeddings are pre-computed externally and loaded from a `.safetensors` file. Tonemapping and EXR saving are the caller's responsibility. LoRA and embeddings: [`Lightricks/LTX-2.3-22b-IC-LoRA-HDR`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR).

**Extra CLI arguments:** `--input` (mp4 or directory, required), `--output-dir` (required), `--hdr-lora` (required), `--text-embeddings` (pre-computed `.safetensors`, required), `--num-frames`, `--spatial-tile` (tiled VAE decode tile size; reduce on lower-VRAM GPUs), `--skip-mp4` (EXR only, no H.264 preview), `--exr-half` (float16 EXR), `--high-quality` (generates 2x frames internally for smoother output, ~2x slower), `--offload {none,cpu,disk}` (weight offloading; disables FP8 quantization when not `none`).

**Use when:** You need linear HDR float output for EXR export, color grading, or custom tonemapping workflows.

---

### 10. LipDubPipeline

**Best for:** Lip dubbing, rephrasing while keeping the same speaker identity and matching lip movements to new audio.

**Source**: [`src/ltx_pipelines/lipdub.py`](src/ltx_pipelines/lipdub.py)

Uses IC-LoRA on a **distilled** checkpoint with a **single** lip-dub IC-LoRA applied in **both** stages. The reference clip provides video and audio reference tokens whose VAE latents are appended to the target audio sequence as frozen reference tokens. The frame count and frame rate are derived from the reference video (frame count is silently snapped to the nearest `8k+1`), so the CLI does not accept `--num-frames` or `--frame-rate`. Required: `--reference-video`. Optional: `--reference-strength`. LoRA: [`Lightricks/LTX-2.3-22b-IC-LoRA-LipDub`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub).

**Note:** Requires a distilled model checkpoint and one lip-dub IC-LoRA (`--lora` exactly once).

**Use when:** Dubbing, rephrasing with matched lips and speaker identity.

---

### 11. T2AOneStagePipeline

**Best for:** Text-to-audio — generating speech/audio only (no video) from a text prompt, e.g. driving an audio-style LoRA such as an accent LoRA.

**Source**: [`src/ltx_pipelines/t2a_one_stage.py`](src/ltx_pipelines/t2a_one_stage.py)

Single-stage, **audio-only** generation: the video branch is absent (`video=None`), so only the audio modality is denoised and decoded through the audio VAE + vocoder, producing a wave file. Audio duration is derived from `--num-frames` / `--frame-rate` (the same `8k+1` frame convention as video). Audio guidance (CFG/STG) is optional — the `--audio-*` flags default to the model's values; the video→audio cross-modal guidance is disabled since there is no video modality.

**Extra CLI arguments (all optional, with sensible defaults):** `--num-frames`, `--frame-rate`, `--negative-prompt`, `--audio-cfg-guidance-scale`, `--audio-stg-guidance-scale`, `--audio-stg-blocks`, `--audio-rescale-scale`, `--audio-skip-step`. No `--height/--width/--image` (audio has no spatial dimensions).

**Use when:** You need speech/audio from text alone, or to evaluate an audio-only LoRA (accent, voice style) without generating video.

---

## 🎨 Conditioning Types

Pipelines use different conditioning methods from [`ltx-core`](../ltx-core/) for controlling generation. See the [ltx-core conditioning documentation](../ltx-core/README.md#conditioning--control) for details.

### Image Conditioning

All pipelines support image conditioning, but with different methods:

- **Replacing Latents** ([`image_conditionings_by_replacing_latent`](src/ltx_pipelines/utils/helpers.py)):
  - Used by: `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `DistilledPipeline`, `ICLoraPipeline`
  - Replaces the latent at a specific frame with the encoded image
  - Strong control over specific frames

- **Guiding Latents** ([`image_conditionings_by_adding_guiding_latent`](src/ltx_pipelines/utils/helpers.py)):
  - Used by: `KeyframeInterpolationPipeline`
  - Adds the image as a guiding signal rather than replacing
  - Better for smooth interpolation between keyframes

### Video Conditioning

- **Video Conditioning** (ICLoraPipeline only):
  - Conditions on entire reference videos
  - Useful for video-to-video transformations
  - Uses `VideoConditionByKeyframeIndex` from [`ltx-core`](../ltx-core/)

---

## 🎛️ Multimodal Guidance

LTX-2 pipelines use **multimodal guidance** to steer the diffusion process for both video and audio modalities. Each modality (video, audio) has its own guider with independent parameters, allowing fine-grained control over generation quality and adherence to prompts.

### Guidance Parameters

The `MultiModalGuiderParams` dataclass controls guidance behavior:

| Parameter | Description |
| --------- | ----------- |
| `cfg_scale` | **Classifier-Free Guidance** scale. Higher values make the output adhere more strongly to the text prompt. Typical values: 2.0–5.0. Set to **1.0** to disable. |
| `stg_scale` | **Spatio-Temporal Guidance** scale. Controls perturbation-based guidance for improved temporal coherence. Typical values: 0.5–1.5. Set to **0.0** to disable. |
| `stg_blocks` | Which transformer blocks to perturb for STG (e.g., `[29]` for the last block). Set to **`[]`** to disable STG. |
| `rescale_scale` | Rescales the guided prediction to match the variance of the conditional prediction. Helps prevent over-saturation. Typical values: 0.5–0.7. Set to **0.0** to disable. |
| `modality_scale` | **Modality CFG** scale. Steers the model away from unsynced video and audio results, improving audio-visual coherence. Set to **1.0** to disable. |
| `skip_step` | Skip guidance every N steps. Can speed up inference with minimal quality loss. Set to **0** to disable (never skip). |

### How It Works

The multimodal guider combines three guidance signals during each denoising step:

1. **CFG (Text Guidance)**: Steers generation toward the text prompt by computing `(cond - uncond_text)`.
2. **STG (Perturbation Guidance)**: Improves structural coherence by perturbing specific transformer blocks and steering away from the perturbed prediction.
3. **Modality CFG**: For joint audio-video generation, steers the model away from unsynced video and audio results.

### Example Configuration

```python
from ltx_core.components.guiders import MultiModalGuiderParams

# Video guider: moderate CFG, STG enabled, modality isolation
video_guider_params = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    stg_blocks=[29],
)

# Audio guider: higher CFG for stronger prompt adherence
audio_guider_params = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    stg_blocks=[29],
)
```

> **Tip:** Start with the default values from [`constants.py`](src/ltx_pipelines/utils/constants.py) and adjust based on your use case. Higher `cfg_scale` = stronger prompt adherence but potentially less natural motion; higher `stg_scale` = better temporal coherence but slower inference (requires extra forward passes).
>
> **Tip:** When generating video with audio, set `modality_scale` > 1.0 (e.g., 3.0) to improve audio-visual sync. If generating video-only, set it to 1.0 to disable.

---

## ⚡ Optimization Tips


### Memory Optimization

**FP8 Quantization (Lower Memory Footprint):**

For smaller GPU memory footprint, use the `--quantization` flag and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Two quantization policies are available:

| Policy | CLI Flag | Description |
| ------ | -------- | ----------- |
| **FP8 Cast** | `--quantization fp8-cast` | Downcasts transformer linear weights to FP8 during loading; upcasts on the fly during inference. No extra dependencies. |
| **FP8 Scaled MM** | `--quantization fp8-scaled-mm` | Uses FP8 scaled matrix multiplication via TensorRT-LLM (`tensorrt_llm` must be installed). Best performance on Hopper GPUs. |

**CLI:**

```bash
# FP8 Cast (works on any GPU with FP8 support)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-cast --checkpoint-path=...

# FP8 Scaled MM (requires tensorrt_llm, best on Hopper GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-scaled-mm --checkpoint-path=...
```

**Programmatically:**

When authoring custom scripts, pass a `QuantizationPolicy` to pipeline classes:

```python
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
# Alternative:
# from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_mm_policy

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=ltx_model_path,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=upsampler_path,
    gemma_root=gemma_root_path,
    loras=[],
    quantization=build_fp8_cast_policy(ltx_model_path),
)
pipeline(...)
```

You still need to use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when launching:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python my_denoising_pipeline.py
```

**Memory Cleanup Between Stages:**

By default, pipelines clean GPU memory (especially transformer weights) between stages. If you have enough memory, you can skip this cleanup to reduce running time:

```python
# In pipeline implementations, memory cleanup happens automatically
# between stages. For custom pipelines, you can skip:
# utils.cleanup_memory()  # Comment out if you have enough VRAM
```

### Compilation (`torch.compile`)

Compiling the transformer blocks with `torch.compile` speeds up inference. It is **opt-in and off by default**. The blocks are compiled shape-polymorphically (the sequence dimension is marked dynamic), so one compiled artifact serves any token count without recompiling.

**CLI** — the `--compile` flag maps directly to `CompilationConfig`:

| Form | Result |
| ---- | ------ |
| *(flag absent)* | eager, no compilation |
| `--compile` | compile with defaults |
| `--compile KEY=VALUE ...` | compile, overriding individual fields |

```bash
# Defaults
python -m ltx_pipelines.ti2vid_two_stages --compile --checkpoint-path=...

# reduce-overhead captures CUDA graphs -- the main latency lever for the denoising loop.
# Off by default because graph capture reserves static memory pools (extra VRAM), so it
# trades memory for speed; enable it when you have headroom.
python -m ltx_pipelines.ti2vid_two_stages --compile mode=reduce-overhead --checkpoint-path=...

# Several overrides at once
python -m ltx_pipelines.ti2vid_two_stages \
    --compile mode=max-autotune fullgraph=true dynamic=true --checkpoint-path=...
```

| Field | Values | Default | Notes |
| ----- | ------ | ------- | ----- |
| `mode` | `none`, `reduce-overhead`, `max-autotune`, … | `none` | `reduce-overhead`/`max-autotune` enable CUDA graphs |
| `backend` | `inductor`, `eager`, … | `inductor` | |
| `fullgraph` | `true`/`false` | `false` | |
| `dynamic` | `auto`/`true`/`false` | `auto` | the seq dim is marked dynamic regardless |
| `inductor_config` | JSON object or path to a `.json` | `{}` | `torch._inductor.config` overrides |
| `dynamo_config` | JSON object or path to a `.json` | `{"inline_inbuilt_nn_modules": true, "cache_size_limit": 256}` | `torch._dynamo.config` overrides |

**Controlling inductor / dynamo configs.** `inductor_config` and `dynamo_config` take either an inline JSON object or a path to a `.json` file, applied via `torch._inductor.config.patch(...)` / `torch._dynamo.config.patch(...)` around the compiled forward. They **replace the defaults wholesale — they do not merge**, so when overriding `dynamo_config` re-include any defaults you want to keep:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"max_autotune": true}' \
              'dynamo_config={"inline_inbuilt_nn_modules": true, "cache_size_limit": 256, "recompile_limit": 32}' \
    --checkpoint-path=...
```

**Programmatically**, pass a `CompilationConfig` to the pipeline:

```python
from ltx_core.model.transformer.compiling import CompilationConfig

pipeline = TI2VidTwoStagesPipeline(
    ...,
    compilation_config=CompilationConfig(mode="reduce-overhead"),
)
```

**Faster cache loads: `unsafe_skip_cache_dynamic_shape_guards` (unsafe, opt-in).** Inductor's FX-graph cache re-checks the dynamic-shape guards stored with each entry on every lookup. Setting this flag skips that re-check (every entry is treated as a guard hit), which speeds up warm and cross-process cache loads. It is **not enabled by default** because it is a correctness hazard: a kernel first compiled at a small sequence length keeps int32 address arithmetic, and reusing it at a larger sequence length (roughly **>58k tokens/rank**) overflows int32 and reads out of bounds — surfacing as a CUDA illegal memory access or silently corrupted output. Only enable it when your token counts stay within the range the cached kernels were compiled for:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"unsafe_skip_cache_dynamic_shape_guards": true}' \
    --checkpoint-path=...
```

### Denoising Loop Optimization

**Gradient Estimation Denoising Loop:**

Instead of the standard Euler denoising loop, you can use gradient estimation for fewer steps (~20-30 instead of 40):

```python
from ltx_pipelines.utils import gradient_estimating_euler_denoising_loop

# Use gradient estimation denoising loop
def denoising_loop(sigmas, video_state, audio_state, stepper):
    return gradient_estimating_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        transformer=transformer,
        denoiser=denoiser,
        ge_gamma=2.0,  # Gradient estimation coefficient
    )
```

This allows you to use **20-30 steps instead of 40** while maintaining quality. The gradient estimation function is defined in [`samplers.py`](src/ltx_pipelines/utils/samplers.py).

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory
- **Spatial Upscaler** - Required for two-stage pipelines (except one-stage)
- **Distilled LoRA** - Required for two-stage pipelines (except one-stage and distilled)

---

## 📖 Example: Image-to-Video

```python
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

distilled_lora = [
    LoraPathStrengthAndSDOps(
        "/path/to/distilled_lora.safetensors",
        0.6,
        LTXV_LORA_COMFY_RENAMING_MAP,
    ),
]

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path="/path/to/checkpoint.safetensors",
    distilled_lora=distilled_lora,
    spatial_upsampler_path="/path/to/upsampler.safetensors",
    gemma_root="/path/to/gemma",
    loras=[],
)

video_guider_params = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[29],
)

audio_guider_params = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[29],
)

# Generate video from image. The pipeline returns (video_iterator, audio);
# the caller is responsible for encoding to file via encode_video().
num_frames = 121
frame_rate = 25.0
tiling_config = TilingConfig.default()
video, audio = pipeline(
    prompt="A serene landscape with mountains in the background",
    negative_prompt="worst quality, low quality, blurry, distorted",
    seed=42,
    height=512,
    width=768,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=40,
    video_guider_params=video_guider_params,
    audio_guider_params=audio_guider_params,
    images=[ImageConditioningInput("input_image.jpg", 0, 1.0, 33)],  # path, frame_idx=0, strength=1.0, crf=33
    tiling_config=tiling_config,
)
encode_video(
    video=video,
    fps=frame_rate,
    audio=audio,
    output_path="output.mp4",
    video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
)
```

---

## 🔗 Related Projects

- **[LTX-Core](../ltx-core/)** - Core model implementation and inference components (schedulers, guiders, noisers, patchifiers)
- **[LTX-Trainer](../ltx-trainer/)** - Training and fine-tuning tools
