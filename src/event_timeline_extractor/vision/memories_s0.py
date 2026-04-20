"""Visual frame analysis using memories-s0 (Memories-ai/security_model).

Apache 2.0 licensed Vision-Language Model from memories.ai that runs fully
locally via HuggingFace transformers — no API key required.

Install extras:  pip install -e ".[vision]"
Requires:        GPU with ~6 GB VRAM (CUDA) for reasonable speed; CPU fallback is slow.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_ID = "Memories-ai/security_model"
_DESCRIBE_PROMPT = "Describe what is visually happening in this scene. Be concise (1-2 sentences)."


@dataclass
class FrameDescription:
    timestamp: float  # seconds from start of video
    description: str


class MemoriesS0Analyzer:
    """Lazy-loaded wrapper around the memories-s0 Vision-Language Model.

    The model and processor are downloaded from HuggingFace on first call to
    ``analyze()`` and cached for the lifetime of this instance.
    """

    def __init__(self, *, device: str | None = None) -> None:
        """
        Args:
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect CUDA).
        """
        self._device = self._resolve_device(device)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame_paths: list[str | Path],
        timestamps: list[float],
    ) -> list[FrameDescription]:
        """Run vision inference on each frame and return descriptions.

        Args:
            frame_paths: Paths to JPEG/PNG frames (same length as ``timestamps``).
            timestamps:  Seconds from video start corresponding to each frame.

        Returns:
            List of :class:`FrameDescription` objects in timestamp order.
        """
        if len(frame_paths) != len(timestamps):
            raise ValueError(
                f"frame_paths ({len(frame_paths)}) and timestamps ({len(timestamps)}) must match."
            )
        if not frame_paths:
            return []

        self._ensure_loaded()
        results: list[FrameDescription] = []
        for path, ts in zip(frame_paths, timestamps, strict=True):
            desc = self._describe_frame(Path(path))
            results.append(FrameDescription(timestamp=ts, description=desc))
            logger.debug("Frame %.1fs → %s", ts, desc[:80])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is not None:
            return device
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        warnings.warn(
            "No CUDA device found — memories-s0 will run on CPU (expect slow inference). "
            "A GPU with ~6 GB VRAM is recommended.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The 'vision' extras are required for frame analysis. "
                'Install them with:  pip install -e ".[vision]"'
            ) from e

        logger.info("Loading memories-s0 from HuggingFace (%s) on %s…", _MODEL_ID, self._device)
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self._device,
        )
        self._model.eval()
        logger.info("memories-s0 loaded.")

    def _describe_frame(self, path: Path) -> str:
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Pillow is required for frame analysis. "
                'Install it with:  pip install -e ".[vision]"'
            ) from e

        import torch  # noqa: PLC0415

        image = Image.open(path).convert("RGB")
        inputs = self._processor(
            images=image,
            text=_DESCRIBE_PROMPT,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )

        # Decode only the newly generated tokens (skip the prompt tokens).
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        description = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        return description or "(no description)"
