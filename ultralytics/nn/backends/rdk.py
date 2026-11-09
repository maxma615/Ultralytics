# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-Robotics RDK X5 inference backend for Ultralytics schema-1 RDK exports.

Loads a `{stem}_rdk_model/` deployment directory (compiled `.bin` + `metadata.yaml` from
`format="rdk"` export), validates the actual model I/O against the recorded `metadata['rdk']`
schema-1 contract, and restores the standard Ultralytics prediction tensors from the raw BPU
outputs:

- On Linux x86_64 the model runs through the external `rdkx5-sim` CLI of the OpenExplore
  toolchain (one isolated subprocess per call; never installed automatically).
- On RDK X5 (Linux aarch64) the preinstalled board `hbm_runtime.HB_HBMRuntime` SDK is used.

Decoding (`decode_rdk_outputs`) is pure PyTorch/NumPy and callable on real raw arrays without
any hardware or session; the head math mirrors `ultralytics/nn/modules/head.py` exactly (see
`ultralytics/utils/export/rdk.py` for the raw-output contract).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER, YAML
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

from .base import BaseBackend

TOOLCHAIN_DOC = "https://developer.d-robotics.cc/rdk_x5"
_RE_EXPORT = "Re-export with YOLO(model).export(format='rdk') so the deployment carries a schema-1 'rdk' contract."
_DESCRIBE_TIMEOUT = 600  # seconds for one rdkx5-sim --describe call
_RUN_TIMEOUT = 600  # seconds for one rdkx5-sim run call (the native timeout stays the CLI's own)

_TASKS = ("detect", "segment", "pose", "obb", "classify")  # canonical task order
_TASK_ROLES = {"detect": (), "segment": ("mask",), "pose": ("kpt",), "obb": ("angle",)}  # beyond cls/box
_KPT_DECODE = {"Pose": (2.0, 0.5), "Pose26": (1.0, 0.0)}  # xy = (raw * scale + anchors - offset) * stride
_OBB_HEADS = {"OBB", "OBB26"}  # OBB applies (sigmoid - 0.25) * pi; OBB26 deploys the raw angle


def _check(condition, message):
    """Raise a ValueError with an actionable RDK message when the condition fails.

    Args:
        condition (bool): Contract condition to enforce.
        message (str): Error text describing the mismatch and how to resolve it.

    Raises:
        ValueError: If the condition is False.
    """
    if not condition:
        raise ValueError(f"D-Robotics RDK: {message}")


def _shape_deployment_matches(actual, recorded):
    """Compare an actual deployed shape with the recorded contract shape, strictly.

    Follows the export-time `_verify_compiled` rule: the ordered dimensions must be equal, or
    the recorded shape must be an ordered prefix of the deployed one with the same element
    count (i.e. the deployment only appends trailing unit dimensions for rank promotion). A
    same-element-count permutation is a different layout and never matches.

    Args:
        actual (list[int]): Deployed tensor shape as reported by the actual runtime.
        recorded (list[int]): Recorded contract shape from `metadata['rdk']['outputs']`.

    Returns:
        (bool): Whether the deployed shape is the recorded shape up to trailing unit dims.
    """
    recorded = [int(d) for d in recorded]
    return [int(d) for d in actual] == recorded or (
        len(actual) > len(recorded)
        and [int(d) for d in actual[: len(recorded)]] == recorded
        and int(np.prod(actual)) == int(np.prod(recorded))
    )


def _strides_prove_c_order(strides, shape, itemsize=4):
    """Return whether byte strides prove a dense C-order layout over the reported shape.

    The runtime's byte strides are the authoritative memory contract: when the last axis steps
    one element and every outer axis steps exactly `shape[i+1] * strides[i+1]`, the bytes are
    C-order over the reported shape regardless of the descriptor's layout label (the vendor
    toolchain labels some permuted outputs `NCHW` while its own strides prove C-order).

    Args:
        strides (list[int] | None): Per-axis byte strides as reported by the runtime.
        shape (list[int]): Reported tensor shape the strides step over.
        itemsize (int): Element size in bytes the strides are measured in.

    Returns:
        (bool): True when the strides are dense C-order over `shape`.
    """
    shape = [int(d) for d in shape]
    if not strides or len(strides) != len(shape):
        return False
    units = [int(s) // itemsize for s in strides]
    if any(int(s) % itemsize for s in strides) or units[-1] != 1:
        return False
    expected = 1
    for size, stride in zip(reversed(shape), reversed(units)):
        if stride != expected:
            return False
        expected *= size
    return True


def _write_npz(path, arrays):
    """Write named arrays to a standard uncompressed NPZ file.

    Uses `zipfile` plus NumPy's public NPY writer instead of `numpy.savez(**arrays)`, so any
    legitimate tensor name round-trips verbatim (including keyword-collision names).

    Args:
        path (str | Path): Output NPZ file path.
        arrays (dict[str, np.ndarray]): Named numeric arrays in model order.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)


def _rgb_to_nv12(image):
    """Convert one RGB uint8 image to the packed NV12 layout the compiled X5 model consumes.

    Args:
        image (np.ndarray): uint8 array of shape (H, W, 3) RGB with even H and W.

    Returns:
        (np.ndarray): Contiguous uint8 array of shape (H + H//2, W): Y plane then interleaved UV.

    Raises:
        ValueError: If the image is not RGB uint8 with even dimensions.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"RDK X5 input conversion needs an RGB uint8 (H, W, 3) image, got {image.shape} {image.dtype}."
        )
    h, w = image.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"RDK X5 NV12 input needs even image dimensions, got {h}x{w}.")
    planar = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(-1)
    y, u, v = planar[: h * w], planar[h * w : h * w * 5 // 4], planar[h * w * 5 // 4 :]
    nv12 = np.concatenate((y, np.stack((u, v), -1).reshape(-1)))
    return np.ascontiguousarray(nv12, dtype=np.uint8).reshape(h * 3 // 2, w)


def decode_rdk_outputs(outputs, description, nc):
    """Decode raw RDK BPU outputs into the standard Ultralytics prediction tensors.

    Pure function: no runtime, session or hardware is touched, so it runs on any raw arrays
    matching the recorded contract (floating reference arrays or real BPU outputs alike).
    Batches larger than 1 are supported whenever the raw arrays carry them.

    Args:
        outputs (dict[str, np.ndarray] | list[np.ndarray]): Raw outputs bound by name (mapping)
            or in the recorded `description["outputs"]` order (list).
        description (dict): `metadata['rdk']` schema-1 contract (task, head_type, strides,
            reg_max, branch, end2end, max_det, agnostic_nms, kpt_shape, outputs).
        nc (int): Number of classes, cross-checked against the recorded cls channels.

    Returns:
        (torch.Tensor | list[torch.Tensor]): Standard predictions — detect/pose one2many
            `(B, 4 + nc + extra, N)` (xywh boxes in input pixels, sigmoid scores, raw mask
            coefficients / decoded keypoints / activated angle appended); one2one
            `(B, K, 6 + extra)` top-k rows `[x1, y1, x2, y2, score, class, extra]` with OBB
            boxes staying xywh + angle; segment additionally the raw `(B, nm, H/4, W/4)`
            NCHW prototypes; classify `(B, nc)` softmax probabilities.

    Raises:
        ValueError: If the outputs do not match the recorded contract.
    """
    named = outputs if isinstance(outputs, dict) else {o["name"]: a for o, a in zip(description["outputs"], outputs)}
    task = description["task"]

    if task == "classify":
        logits = torch.as_tensor(np.asarray(named[description["outputs"][0]["name"]]))
        _check(
            logits.ndim >= 2 and logits.shape[-1] == nc, f"classify logits {tuple(logits.shape)} do not end in nc={nc}."
        )
        return logits.reshape(-1, nc).float().softmax(1)  # softmax applied exactly once, host-side

    roles = {(o["role"], o.get("stride")): o["name"] for o in description["outputs"]}
    reg_max = int(description["reg_max"] or 0)
    _check(reg_max >= 1, "the contract records no reg_max.")
    end2end = description["branch"] == "one2one"

    levels = []
    for stride in (int(s) for s in description["strides"]):
        entry = {}
        for role in ("cls", "box", *_TASK_ROLES[task]):
            name = roles.get((role, stride))
            _check(name is not None and name in named, f"missing the '{role}_{stride}' raw output.")
            entry[role] = torch.as_tensor(np.asarray(named[name])).float()
            _check(entry[role].ndim == 4, f"output '{name}' rank {entry[role].ndim} is not the recorded NHWC layout.")
        h, w = (int(d) for d in entry["cls"].shape[1:3])
        _check(
            all(tuple(e.shape[1:3]) == (h, w) for e in entry.values()),
            f"stride-{stride} outputs disagree on the feature size.",
        )
        levels.append(
            (stride, h, w, {r: e.permute(0, 3, 1, 2).reshape(e.shape[0], -1, h * w) for r, e in entry.items()})
        )
    batch = levels[0][3]["cls"].shape[0]

    boxes = torch.cat([lv[3]["box"] for lv in levels], 2)  # (B, 4*reg_max, N) in head level order
    scores = torch.cat([lv[3]["cls"] for lv in levels], 2)  # (B, nc, N)
    _check(scores.shape[1] == nc, f"cls channels {scores.shape[1]} do not match nc={nc}.")
    _check(boxes.shape[1] == 4 * reg_max, f"box channels {boxes.shape[1]} do not match 4*reg_max={4 * reg_max}.")
    anchors, strides_t = make_anchors(  # grid cell centers + per-anchor strides, as in Detect._get_decode_boxes
        torch.tensor([[lv[1], lv[2]] for lv in levels]), torch.tensor([lv[0] for lv in levels]), 0.5
    )
    anchors, strides_t = anchors.T.unsqueeze(0), strides_t.T  # (1, 2, N), (1, N)

    if reg_max > 1:  # DFL: softmax over each side's bins then expectation, like nn.modules.block.DFL
        bins = torch.arange(reg_max, dtype=boxes.dtype).view(1, 1, reg_max, 1)
        boxes = (boxes.view(batch, 4, reg_max, -1).softmax(2) * bins).sum(2)
    scores = scores.sigmoid()  # full per-class probabilities; thresholds stay with the upper owner
    extras = []
    if task == "segment":
        extras.append(torch.cat([lv[3]["mask"] for lv in levels], 2))  # raw coefficients, no activation
    elif task == "pose":
        nkpt, ndim = (int(k) for k in description["kpt_shape"])
        scale, offset = _KPT_DECODE[description["head_type"]]
        kpts = torch.cat([lv[3]["kpt"] for lv in levels], 2)  # (B, nkpt*ndim, N)
        _check(kpts.shape[1] == nkpt * ndim, f"kpt channels {kpts.shape[1]} do not match kpt_shape {(nkpt, ndim)}.")
        kpts = kpts.view(batch, nkpt, ndim, -1)
        xy = (kpts[:, :, :2] * scale + anchors - offset) * strides_t
        if ndim == 3:
            xy = torch.cat((xy, kpts[:, :, 2:3].sigmoid()), 2)  # visibility sigmoid exactly once
        extras.append(xy.reshape(batch, nkpt * ndim, -1))
    elif task == "obb":
        angle = torch.cat([lv[3]["angle"] for lv in levels], 2)  # (B, ne, N)
        _check(angle.shape[1] == 1, f"obb angle channels {angle.shape[1]} do not match the single recorded channel.")
        if description["head_type"] == "OBB":  # head.py OBB.forward_head activation, decode-side here
            angle = (angle.sigmoid() - 0.25) * torch.pi
        boxes = dist2rbox(boxes, angle, anchors, dim=1) * strides_t  # rotated xywh, never xyxy
        extras.append(angle)
    if task != "obb":
        boxes = dist2bbox(boxes, anchors, xywh=not end2end, dim=1) * strides_t  # xywh one2many, xyxy one2one
    preds = torch.cat((boxes, scores, *extras), 1)  # (B, 4 + nc + extra, N)

    if end2end:  # one2one: the head's own top-k selection (Detect.postprocess/get_topk_index)
        k = min(int(description["max_det"]), preds.shape[-1])
        rows = preds.permute(0, 2, 1)  # (B, N, C)
        box, score, *extra = rows.split([4, nc, rows.shape[-1] - 4 - nc], dim=-1)

        def gather(x, i):  # select the (batch, k) index rows of x along dim 1, like Detect._gather
            return x.gather(1, i[..., None].expand(-1, -1, x.shape[-1]))

        if description["agnostic_nms"]:  # class-agnostic: max over classes, then top-k anchors
            score, label = score.max(-1)
            value, index = score.topk(k, 1)
            label = label.gather(1, index).float()[..., None]
        else:  # two-stage: top anchors by max score, then top (anchor, class) pairs
            ori = score.max(-1)[0].topk(k, 1)[1]
            value, index = gather(score, ori).flatten(1).topk(k, 1)
            label = (index % nc).float()[..., None]
            index = ori.gather(1, index // nc)
        preds = torch.cat((gather(box, index), value[..., None], label, *(gather(e, index) for e in extra)), -1)

    if task == "segment":
        proto = torch.as_tensor(np.asarray(named[roles[("proto", 4)]])).permute(0, 3, 1, 2)  # NHWC -> NCHW
        return [preds, proto.contiguous().float()]
    return preds


class RDKBackend(BaseBackend):
    """D-Robotics RDK X5 inference backend for `.bin` models compiled by `hb_mapper`.

    Loads a schema-1 Ultralytics RDK deployment (one compiled `.bin` plus its `metadata.yaml`),
    verifies the actual runtime I/O against the recorded contract, converts RGB uint8 input to
    the model's packed NV12 input and decodes the raw outputs back to standard predictions.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an Ultralytics RDK X5 deployment directory or its compiled .bin file.

        Args:
            weight (str | Path): `{stem}_rdk_model/` directory, or the `.bin` inside one (a
                bare `.bin` without a sibling `metadata.yaml` carrying a schema-1 `rdk` contract
                is rejected — RDK never guesses a generic `.bin` layout).

        Raises:
            (ValueError | FileNotFoundError | RuntimeError): If the deployment is ambiguous,
                the recorded contract is invalid, or the actual runtime I/O disagrees with it.
        """
        w = Path(weight)
        if w.is_dir():
            bins = sorted(w.glob("*.bin"))  # top level only; never a recursive first-match
            _check(
                len(bins) == 1,
                f"'{w}' contains {len(bins)} top-level .bin files {sorted(p.name for p in bins)}; point at the exact "
                f"'{{stem}}_rdk_model' export directory or .bin file. {_RE_EXPORT}",
            )
            w = bins[0]
        if not w.is_file():
            raise FileNotFoundError(f"D-Robotics RDK model file not found: '{w}'.")
        sidecar = w.parent / "metadata.yaml"
        _check(
            sidecar.is_file(),
            f"'{w}' has no metadata.yaml with an Ultralytics RDK contract; a bare .bin is not routable. {_RE_EXPORT}",
        )
        metadata = YAML.load(sidecar)
        description = metadata.get("rdk")
        _check(isinstance(description, dict), f"'{sidecar}' carries no 'rdk' contract entry. {_RE_EXPORT}")
        self._validate_contract(description)
        self.apply_metadata(metadata)
        self._bin, self._description = w, description
        self.dynamic = False  # static deployments; AutoBackend serializes larger batches
        _check(
            self.task == description["task"],
            f"metadata task '{self.task}' disagrees with the rdk contract task '{description['task']}'.",
        )
        _check(bool(self.names), "the deployment metadata carries no class names.")
        if LINUX and not ARM64:
            self._load_simulator()
        elif LINUX and ARM64:
            self._load_board_runtime()
        else:
            raise RuntimeError(
                f"D-Robotics RDK X5 inference runs on Linux x86_64 (rdkx5-sim) or RDK boards (hbm_runtime); this "
                f"platform is unsupported. See {TOOLCHAIN_DOC}."
            )
        if "compiled" in description:  # cached export-time specs must still match the actual I/O
            self._validate_runtime_spec({"schema_version": 1, **description["compiled"]}, "cached rdk.compiled spec")
        if description["task"] != "classify":
            _check(
                tuple(int(d) for d in self.imgsz) == (int(self._hw[0]), int(self._hw[1])),
                f"metadata imgsz {self.imgsz} disagrees with the compiled input {list(self._hw)}.",
            )
        LOGGER.info(f"Loaded {w.name} for D-Robotics RDK X5 {self.task} inference.")

    def _validate_contract(self, description):
        """Validate the recorded schema-1 contract before anything is loaded from it.

        Args:
            description (dict): The `metadata['rdk']` entry of the deployment.
        """
        _check(
            description.get("schema_version") == 1,
            f"unsupported rdk schema_version {description.get('schema_version')!r}. {_RE_EXPORT}",
        )
        task = description.get("task")
        _check(task in _TASKS, f"rdk task '{task}' is not one of {_TASKS}.")
        outputs = description.get("outputs")
        _check(isinstance(outputs, list) and outputs, "rdk contract lists no outputs.")
        for output in outputs:
            _check(
                isinstance(output, dict) and {"name", "role", "layout"} <= set(output),
                f"rdk output entry {output!r} lacks name/role/layout.",
            )
            _check(
                output.get("layout") == "NHWC",
                f"output '{output['name']}' layout {output.get('layout')!r} is not NHWC.",
            )
        if task == "classify":
            _check([o["role"] for o in outputs] == ["logits"], "classify contract expects the single 'logits' output.")
            return
        strides = description.get("strides")
        _check(isinstance(strides, list) and len(strides) > 0, "rdk contract records no strides.")
        reg_max = description.get("reg_max")
        _check(isinstance(reg_max, int) and reg_max >= 1, f"rdk contract reg_max {reg_max!r} is invalid.")
        _check(
            description.get("branch") in {"one2many", "one2one"},
            f"rdk branch {description.get('branch')!r} is invalid.",
        )
        _check(
            (description.get("branch") == "one2one") == bool(description.get("end2end")),
            "rdk end2end flag disagrees with the recorded branch.",
        )
        for key in ("max_det", "agnostic_nms"):
            _check(
                description.get(key) is not None, f"rdk contract lacks '{key}'; the export recorded no top-k settings."
            )
        for stride in strides:
            expected = sorted(("cls", "box", *_TASK_ROLES[task]))
            got = sorted(o["role"] for o in outputs if o.get("stride") == stride)
            _check(got == expected, f"stride {stride} records roles {got}, expected {expected}.")
        if task == "segment":
            protos = [o for o in outputs if o["role"] == "proto"]
            _check(
                len(protos) == 1 and protos[0].get("stride") == 4,
                "segment contract lacks its single stride-4 'proto' output.",
            )
        if task == "pose":
            kpt_shape = description.get("kpt_shape")
            _check(
                isinstance(kpt_shape, list) and len(kpt_shape) == 2 and kpt_shape[1] in (2, 3),
                f"pose contract kpt_shape {kpt_shape!r} is invalid.",
            )
            _check(
                description.get("head_type") in _KPT_DECODE,
                f"pose head_type {description.get('head_type')!r} has no known decode. {_RE_EXPORT}",
            )
        if task == "obb":
            _check(
                description.get("head_type") in _OBB_HEADS,
                f"obb head_type {description.get('head_type')!r} has no known decode. {_RE_EXPORT}",
            )

    def _validate_runtime_spec(self, spec, source):
        """Validate an actual runtime I/O descriptor against the recorded contract.

        Args:
            spec (dict): schema-1 descriptor from `rdkx5-sim --describe` or `rdk.compiled`.
            source (str): Provenance of the descriptor, for error messages.
        """
        _check(spec.get("schema_version") == 1, f"{source} returned schema {spec.get('schema_version')!r}, expected 1.")
        inputs = spec.get("inputs")
        _check(
            isinstance(inputs, list) and len(inputs) == 1,
            f"{source} reports {len(inputs) if isinstance(inputs, list) else inputs} inputs; the deployment expects "
            f"the single compiled NV12 input.",
        )
        logical = [int(d) for d in inputs[0]["shape"]]
        h, w = logical[-2:]
        _check(
            logical == [1, 3, h, w] and inputs[0].get("dtype") == "uint8" and inputs[0].get("layout") == "NCHW",
            f"compiled input {inputs[0]} is not the recorded uint8 NCHW (1, 3, H, W) NV12 input.",
        )
        _check(
            inputs[0].get("pixel_format") == "nv12",
            f"compiled input pixel_format {inputs[0].get('pixel_format')!r} is not nv12.",
        )
        _check(h % 2 == 0 and w % 2 == 0, f"compiled input {h}x{w} has odd dimensions; NV12 needs even ones.")
        storage = inputs[0].get("storage_shape")
        _check(
            storage is None or [int(d) for d in storage] == [h + h // 2, w],
            f"compiled NV12 storage_shape {storage} does not match (H + H//2, W) = ({h + h // 2}, {w}).",
        )
        actual = [o["name"] for o in spec.get("outputs", [])]
        expected = [o["name"] for o in self._description["outputs"]]
        _check(
            actual == expected,
            f"{source} reports outputs {actual} but the contract records {expected}; refusing to guess the role "
            f"binding. {_RE_EXPORT}",
        )
        for got, recorded in zip(spec.get("outputs", []), self._description["outputs"]):
            name = recorded["name"]
            shape = [int(d) for d in got["shape"]]
            _check(
                got.get("dtype") == "float32" and got.get("quanti_type") == "NONE",
                f"output '{name}' deploys as {got.get('dtype')}/{got.get('quanti_type')}; only dequantized "
                f"float32 outputs have a known decode — re-export or use the toolchain runtime.",
            )
            # The byte strides are the authoritative layout; the label alone is only trusted
            # when no strides prove otherwise — an NCHW-labeled tensor is never reinterpreted.
            if _strides_prove_c_order(got.get("strides"), shape) and got.get("layout") not in {None, "NHWC"}:
                LOGGER.warning(
                    f"D-Robotics RDK: output '{name}' is labeled {got.get('layout')!r} but its byte strides prove the "
                    f"recorded NHWC C-order layout; using the strides."
                )
            else:
                _check(
                    got.get("layout") is None or got.get("layout") == "NHWC",
                    f"output '{name}' deploys as layout {got.get('layout')!r} with strides {got.get('strides')}; "
                    f"expected the recorded NHWC layout and no proof of it — never reinterpreted. {_RE_EXPORT}",
                )
            _check(
                _shape_deployment_matches(shape, recorded["shape"]),
                f"output '{name}' deploys as {shape}, the contract records {recorded['shape']}; a same-size "
                f"reordering is a different layout and is never reinterpreted. {_RE_EXPORT}",
            )
        self._input_name, self._hw = inputs[0]["name"], (h, w)

    def _load_simulator(self):
        """Resolve the x86 simulator CLI and validate the model's actual I/O through it."""
        sim = shutil.which("rdkx5-sim")
        if sim is None:
            raise RuntimeError(
                "D-Robotics RDK: 'rdkx5-sim' not found on PATH. Install the D-Robotics OpenExplore X5 toolchain "
                f"(Python 3.10) and put its bin directory on PATH, or run on an RDK X5 board. See {TOOLCHAIN_DOC}. "
                "Dependencies are never installed automatically."
            )
        self._sim = sim
        try:
            result = subprocess.run(
                [sim, str(self._bin), "--describe"],
                capture_output=True,
                text=True,
                timeout=_DESCRIBE_TIMEOUT,
                check=False,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"D-Robotics RDK: rdkx5-sim --describe timed out after {_DESCRIBE_TIMEOUT}s.") from None
        if result.returncode != 0:
            raise RuntimeError(
                f"D-Robotics RDK: rdkx5-sim --describe failed with exit code {result.returncode}: "
                f"{result.stderr.strip()[-1000:]}"
            )
        try:
            spec = json.loads(result.stdout)
        except json.JSONDecodeError:
            detail = f"{result.stdout[:300]!r} {result.stderr[:300]!r}"
            raise ValueError(f"D-Robotics RDK: rdkx5-sim --describe returned invalid JSON: {detail}") from None
        self._validate_runtime_spec(spec, "rdkx5-sim --describe")

    def _load_board_runtime(self):
        """Lazily construct the preinstalled board SDK session and validate its actual I/O.

        Validates the SDK's own reported values — never assumed ones — against the recorded
        contract: single NV12 input (dtype, logical shape, no input quantization; the model
        applies its configured input scaling itself) and float32/non-quantized outputs with
        matching names and strictly matching shapes. A cached `rdk.compiled` spec, when
        present, is cross-checked against the same actual SDK values. The SDK exposes names,
        dtypes, quanti types, logical shapes and strides, but no storage shape or per-tensor
        layout string; anything it does not expose is not claimed as verified.
        """
        try:
            import hbm_runtime
        except ImportError as e:
            raise RuntimeError(
                "D-Robotics RDK: hbm_runtime is required on RDK boards but not importable. Use the board's matching "
                f"Python environment (it is never auto-installed). See {TOOLCHAIN_DOC}."
            ) from e
        dt = getattr(hbm_runtime, "hbDNNDataType", None)
        qt = getattr(hbm_runtime, "hbDNNQuantiType", None)
        nv12 = f32 = none = None
        if dt is not None:
            nv12, f32 = getattr(dt, "NV12", None), getattr(dt, "F32", None)
        if qt is not None:
            none = getattr(qt, "NONE", None)
        _check(
            nv12 is not None and f32 is not None and none is not None,
            "the installed hbm_runtime does not expose the hbDNNDataType NV12/F32 or hbDNNQuantiType NONE members "
            "needed to verify the model I/O; use the RDK's matching SDK. "
            "Re-export or load this deployment with the rdkx5-sim toolchain instead.",
        )
        self._runtime = hbm_runtime.HB_HBMRuntime(str(self._bin))
        names = self._runtime.model_names
        _check(len(names) == 1, f"hbm_runtime loaded {len(names)} models {names}; expected exactly one.")
        (model_name,) = names
        input_names = self._runtime.input_names[model_name]
        _check(
            len(input_names) == 1,
            f"board model exposes {len(input_names)} inputs; the deployment expects the single NV12 input.",
        )
        input_name = input_names[0]
        input_dtype = str(self._runtime.input_dtypes[model_name][input_name])
        _check(
            input_dtype == str(nv12),
            f"board input '{input_name}' dtype {input_dtype} is not the NV12 input this backend feeds; only packed "
            f"NV12 deployments are supported. {_RE_EXPORT}",
        )
        logical = [int(d) for d in self._runtime.input_shapes[model_name][input_name]]
        h, w = logical[-2:]
        _check(
            logical == [1, 3, h, w] and h % 2 == 0 and w % 2 == 0,
            f"board input shape {logical} is not an even-dimension (1, 3, H, W) NV12 input.",
        )
        input_quant = self._runtime.input_quants[model_name][input_name].quant_type
        _check(
            str(input_quant) == str(none),
            f"board input '{input_name}' carries input quantization {input_quant}; raw packed NV12 bytes would not be "
            f"interpreted correctly. {_RE_EXPORT}",
        )
        output_names = list(self._runtime.output_names[model_name])
        expected = [o["name"] for o in self._description["outputs"]]
        _check(
            output_names == expected,
            f"board outputs {output_names} do not match the contract {expected}; refusing to guess. {_RE_EXPORT}",
        )
        dtypes = self._runtime.output_dtypes[model_name]
        quants = self._runtime.output_quants[model_name]
        shapes = self._runtime.output_shapes[model_name]
        strides = getattr(self._runtime, "output_strides", {}).get(model_name, {})
        for name, recorded in zip(output_names, self._description["outputs"]):
            _check(
                str(dtypes[name]) == str(f32) and str(quants[name].quant_type) == str(none),
                f"board output '{name}' deploys as {dtypes[name]}/{quants[name].quant_type}; only dequantized float32 "
                f"outputs have a known decode — re-export or use the toolchain runtime.",
            )
            _check(
                _shape_deployment_matches(shapes[name], recorded["shape"]),
                f"board output '{name}' deploys as {shapes[name]}, the contract records {recorded['shape']}; "
                f"a same-size reordering is a different layout and is never reinterpreted. {_RE_EXPORT}",
            )
            _check(
                not strides.get(name) or _strides_prove_c_order(strides[name], shapes[name]),
                f"board output '{name}' byte strides {strides.get(name)} are not dense C-order over "
                f"{shapes[name]}; the recorded NHWC layout is not proven and is never reinterpreted. {_RE_EXPORT}",
            )
        if "compiled" in self._description:  # cached export-time specs must match the actual board I/O
            cached = self._description["compiled"]
            cached_in = cached.get("inputs", [{}])[0]
            _check(
                cached_in.get("name") == input_name and [int(d) for d in cached_in.get("shape", [])] == [1, 3, h, w],
                f"cached rdk.compiled input {cached_in.get('name')}/{cached_in.get('shape')} does not match the actual "
                f"board input '{input_name}'/{logical}. {_RE_EXPORT}",
            )
            for got, name in zip(cached.get("outputs", []), output_names):
                _check(
                    got.get("name") == name
                    and got.get("dtype") == "float32"
                    and _shape_deployment_matches(got.get("shape", []), shapes[name]),
                    f"cached rdk.compiled output {got.get('name')} does not match the actual board output "
                    f"'{name}' ({shapes[name]}). {_RE_EXPORT}",
                )
        self._model_name, self._input_name, self._hw = model_name, input_name, (h, w)

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on one RGB image batch through the RDK X5 model.

        Args:
            im (torch.Tensor): Input tensor in BHWC format (AutoBackend permutes BCHW for this
                nhwc backend), RGB, float 0-1. Batch must be 1: the deployment is static
                batch-1 and `AutoBackend` serializes larger batches through repeated calls.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Standard decoded predictions (see `decode_rdk_outputs`).
        """
        if im.ndim != 4 or im.shape[-1] != 3 or im.shape[0] != 1:
            raise ValueError(
                f"D-Robotics RDK deployments are static batch-1 and take one RGB BHWC image, got {tuple(im.shape)}; "
                f"larger batches are split automatically by AutoBackend/predict/val."
            )
        image = (im.detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)[0]
        if image.shape[:2] != self._hw:
            raise ValueError(
                f"input image {tuple(image.shape[:2])} does not match the compiled model input {list(self._hw)}; "
                f"predict/val letterbox to imgsz={list(self._hw)}."
            )
        nv12 = _rgb_to_nv12(image)
        raw = self._run_simulator(nv12) if hasattr(self, "_sim") else self._run_board(nv12)
        return decode_rdk_outputs(raw, self._description, len(self.names))

    def _run_simulator(self, nv12):
        """Execute one rdkx5-sim run call in a fresh scratch directory and copy the outputs by name."""
        with tempfile.TemporaryDirectory(prefix="ultralytics_rdk_") as td:
            _write_npz(Path(td) / "input.npz", {self._input_name: nv12})
            try:
                result = subprocess.run(
                    [self._sim, str(self._bin), "--inputs", "input.npz", "--outputs", "output.npz"],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=_RUN_TIMEOUT,
                    check=False,
                    env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"D-Robotics RDK: rdkx5-sim timed out after {_RUN_TIMEOUT}s.") from None
            if result.returncode != 0:
                raise RuntimeError(
                    f"D-Robotics RDK: rdkx5-sim failed with exit code {result.returncode}: "
                    f"{result.stderr.strip()[-1000:]}"
                )
            out = Path(td) / "output.npz"
            _check(out.is_file(), f"rdkx5-sim wrote no output file: {result.stderr.strip()[-500:]}")
            with np.load(out, allow_pickle=False) as npz:
                missing = [o["name"] for o in self._description["outputs"] if o["name"] not in npz.files]
                _check(not missing, f"rdkx5-sim outputs {npz.files} miss the contract names {missing}.")
                return {
                    o["name"]: np.array(npz[o["name"]], dtype=np.float32, copy=True)
                    for o in self._description["outputs"]
                }

    def _run_board(self, nv12):
        """Execute one board SDK run and return fresh copies of the borrowed outputs."""
        outputs = self._runtime.run({self._model_name: {self._input_name: nv12.reshape(-1)}})[self._model_name]
        return {
            o["name"]: np.array(outputs[o["name"]], dtype=np.float32, copy=True) for o in self._description["outputs"]
        }
