from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import cv2

from ultralytics.utils import LOGGER
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_rdk_requirements

from .base import BaseBackend


class RDKBackend(BaseBackend):
    """D-Robotics RDK X5 inference backend for .bin models compiled by hb_mapper."""

    def load_model(self, weight: str | Path) -> None:
        """Load an RDK X5 model directory or compiled .bin file."""
        check_rdk_requirements()

        import hbm_runtime

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.bin"))

        LOGGER.info(f"Loading {w} for D-Robotics inference...")
        self.model = hbm_runtime.HB_HBMRuntime(str(w))
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.nhwc = True
        self.input_shape = getattr(self.model, "input_shapes", {}).get(self.model_name, {}).get(self.input_names[0])
        self._nv12_buffer = None
        self._nv12_buffer_size = 0
        self._profile_enabled = os.getenv("ULTRALYTICS_RDK_PROFILE", "0") == "1"
        self._profile_log_interval = max(int(os.getenv("ULTRALYTICS_RDK_PROFILE_INTERVAL", "50")), 1)
        self._profile = {"calls": 0, "preprocess_ms": 0.0, "infer_ms": 0.0, "decode_ms": 0.0}
        if len(self.input_names) != 1:
            raise RuntimeError(
                f"RDK X5 runtime expects a single NV12 input tensor, but model '{w.name}' exposes "
                f"{len(self.input_names)} inputs."
            )

        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            self.apply_metadata(YAML.load(metadata_file))

    def _get_nv12_buffer(self, size: int) -> np.ndarray:
        """Return a reusable flat NV12 buffer."""
        if self._nv12_buffer is None or self._nv12_buffer_size != size:
            self._nv12_buffer = np.empty(size, dtype=np.uint8)
            self._nv12_buffer_size = size
        return self._nv12_buffer

    def _convert_rgb_to_nv12(self, img: np.ndarray) -> np.ndarray:
        """Convert a single RGB image to the NV12 layout expected by the X5 runtime."""
        h, w = img.shape[:2]
        flat_size = h * w * 3 // 2
        yuv420p = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420).reshape((flat_size,))
        y_size = h * w
        uv_size = y_size // 4
        y_arr = yuv420p[:y_size]
        u_arr = yuv420p[y_size : y_size + uv_size]
        v_arr = yuv420p[y_size + uv_size :]

        nv12_flat = self._get_nv12_buffer(flat_size).copy()
        nv12_flat[:y_size] = y_arr
        uv_view = nv12_flat[y_size:].reshape(h // 2, w // 2, 2)
        uv_view[..., 0] = u_arr.reshape(h // 2, w // 2)
        uv_view[..., 1] = v_arr.reshape(h // 2, w // 2)

        if self.input_shape is not None and np.prod(self.input_shape) == flat_size:
            return nv12_flat.reshape(self.input_shape[1:])
        return nv12_flat.reshape(int(h * 1.5), w, 1)

    def _update_profile(self, preprocess_ms: float, infer_ms: float, decode_ms: float) -> None:
        """Accumulate and periodically print lightweight runtime profiling stats."""
        if not self._profile_enabled:
            return
        self._profile["calls"] += 1
        self._profile["preprocess_ms"] += preprocess_ms
        self._profile["infer_ms"] += infer_ms
        self._profile["decode_ms"] += decode_ms
        if self._profile["calls"] % self._profile_log_interval == 0:
            calls = self._profile["calls"]
            LOGGER.info(
                "RDK X5 profile avg over %d runs: preprocess=%.3fms infer=%.3fms decode=%.3fms",
                calls,
                self._profile["preprocess_ms"] / calls,
                self._profile["infer_ms"] / calls,
                self._profile["decode_ms"] / calls,
            )

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on an RDK model."""
        t0 = time.perf_counter()
        im = im.cpu().numpy()
        if im.dtype != np.uint8:
            im = (im * 255.0).clip(0, 255).astype(np.uint8)

        single_nv12s = [self._convert_rgb_to_nv12(img) for img in im]
        nv12_batch = single_nv12s[0][None, ...] if len(single_nv12s) == 1 else np.stack(single_nv12s, axis=0)
        preprocess_ms = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        input_tensor = {self.model_name: {self.input_names[0]: nv12_batch}}
        outputs = self.model.run(input_tensor)[self.model_name]
        infer_ms = (time.perf_counter() - t1) * 1e3

        y = [outputs[name] for name in self.output_names]

        if self.task == "classify":
            self._update_profile(preprocess_ms, infer_ms, 0.0)
            return y[0]

        from ultralytics.utils.export.rdk import decode_rdk

        t2 = time.perf_counter()
        decoded = decode_rdk(y, self.imgsz, nc=len(self.names))
        decode_ms = (time.perf_counter() - t2) * 1e3
        self._update_profile(preprocess_ms, infer_ms, decode_ms)
        return decoded
