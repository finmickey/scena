"""LTX-2 audio-only pipelines (ScenA).

Reference-driven multi-speaker audio scene generation.

- :class:`T2AudRefCondPipeline`: reference-conditioned text-to-audio (ScenA) --
  give one or more reference-audio waveforms plus a text prompt and generate a
  new audio scene in those voices.
- :class:`T2AOneStagePipeline`: single-stage text-to-audio (no reference).
"""

from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
from ltx_pipelines.t2aud_ref_cond import T2AudRefCondPipeline

__all__ = [
    "T2AOneStagePipeline",
    "T2AudRefCondPipeline",
]
