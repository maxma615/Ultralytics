# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-Robotics RDK X5 BPU export logic.

Raw-head contract (consumed by the RDK backend, ``ultralytics/nn/backends/rdk.py``): every head exports its active
(one2one or one2many) branch as one NHWC tensor per scale, named ``<role>_<stride>`` (plus ``proto`` for Segment and
``logits`` for Classify), recorded in ``metadata['rdk']['outputs']`` and bound by name at runtime.

Roles and their numerical meaning (``anchors`` = grid cell centers from ``make_anchors(..., grid_cell_offset=0.5)``
= integer grid + 0.5; the recorded ``reg_max`` decides the box path, not the model family):

- ``cls``: raw class logits per anchor; sigmoid gives per-class object scores.
- ``box``: ``reg_max > 1``: ``4 * reg_max`` DFL bin logits, host applies softmax over each side's bins, expectation,
  then ``(anchors ± ltrb) * stride``; ``reg_max == 1``: direct LTRB distances in stride units with the same anchor
  equation.
- ``mask``: raw mask coefficients (``nm``) per anchor; no activation; the host combines them with the prototypes
  the way Ultralytics ``process_mask`` does.
- ``proto``: prototype map shaped ``(1, H/4, W/4, nm)`` (P3 feature upsampled 2x); no activation.
- ``kpt``: raw keypoints (``nk`` channels). Pose (YOLO11): ``(raw_xy * 2 + (anchors - 0.5)) * stride`` — i.e.
  ``(raw_xy * 2 + integer_grid) * stride`` — and ``sigmoid(raw_v)``; Pose26: ``(raw_xy + anchors) * stride`` (no
  2x scale) and ``sigmoid(raw_v)``.
- ``angle``: OBB11: ``(sigmoid(raw) - 0.25) * pi`` in ``[-pi/4, 3pi/4]``; OBB26: raw angle in radians. Boxes decode
  with ``dist2rbox(ltrb, angle, anchors)``.
- ``logits``: classifier logits; softmax is applied exactly once on the host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER, YAML, colorstr

TOOLCHAIN_DOC = "https://developer.d-robotics.cc/rdk_x5"
_COMPILE_TIMEOUT = 3600  # seconds; a 640px O3 INT8 compile of a nano model takes minutes
_DESCRIBE_TIMEOUT = 600  # seconds for one rdkx5-sim --describe call


def _rdk_branch(head):
    """Return the fused head's active component dict, failing clearly when the branch is incomplete.

    Args:
        head (Detect): Fused YOLO detect-family head (the exporter already selected and fused one branch).

    Returns:
        (dict): The head's ``one2one`` or ``one2many`` component dict with every inference component non-None.

    Raises:
        ValueError: If the selected branch is missing inference components (e.g. fusion removed both heads).
    """
    branch = head.one2one if head.end2end else head.one2many
    required = [k for k in branch if k != "kpts_sigma_head"]  # sigma is a training-only output
    missing = [k for k in required if branch[k] is None]
    if missing:
        raise ValueError(
            f"D-Robotics RDK export needs a complete '{'one2one' if head.end2end else 'one2many'}' head, but "
            f"{type(head).__name__} is missing {missing}. Re-export from the original (unfused) checkpoint."
        )
    return branch


def _rdk_detect_forward(self, x):
    """Detect-family raw BPU forward: cls + box NHWC tensors per scale from the active branch."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
    return res


def _rdk_segment_forward(self, x):
    """YOLO11 Segment raw BPU forward: cls + box + mask NHWC per scale, then the P3 prototype map."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
        res.append(branch["mask_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # mask coefficients
    res.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())  # protos, H/4 x W/4
    return res


def _rdk_segment26_forward(self, x):
    """YOLO26 Segment raw BPU forward; Proto26 consumes the whole feature list, unlike Proto."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
        res.append(branch["mask_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # mask coefficients
    res.append(self.proto(x).permute(0, 2, 3, 1).contiguous())  # protos from the fused multi-scale features
    return res


def _rdk_pose_forward(self, x):
    """YOLO11 Pose raw BPU forward: cls + box + kpt (nk channels) NHWC per scale."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
        res.append(branch["pose_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # keypoints
    return res


def _rdk_pose26_forward(self, x):
    """YOLO26 Pose raw BPU forward; keypoints come from cv4_kpts applied to the cv4 feature maps."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        feature = branch["pose_head"][i](x[i])  # cv4 feature extractor
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
        res.append(branch["kpts_head"][i](feature).permute(0, 2, 3, 1).contiguous())  # nk keypoint channels
    return res


def _rdk_obb_forward(self, x):
    """OBB raw BPU forward: cls + box + angle NHWC per scale; the angle activation is decode-side."""
    branch = _rdk_branch(self)
    res = []
    for i in range(self.nl):
        res.append(branch["cls_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(branch["box_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # box
        res.append(branch["angle_head"][i](x[i]).permute(0, 2, 3, 1).contiguous())  # raw angle
    return res


def _rdk_classify_forward(self, x):
    """Classify raw BPU forward: NHWC logits (1, 1, 1, nc); the host applies softmax exactly once."""
    if isinstance(x, list):
        x = torch.cat(x, 1)
    return self.linear(self.pool(self.conv(x))).permute(0, 2, 3, 1).contiguous()  # linear is a 1x1 Conv2d here


def apply_rdk_patches(model):
    """Bind raw BPU head forwards on the exporter's model copy and return its `rdk` metadata description.

    Args:
        model (torch.nn.Module): The exporter's deep copy of a fused YOLO model with one selected branch.

    Returns:
        (dict): `rdk` schema-1 metadata (task, head_type, strides, reg_max, branch, kpt_shape and the ordered
            `outputs` role list; the exporter fills each entry's `shape` from its dry run).

    Raises:
        TypeError | ValueError: If the model's head is not one of the standard YOLO11/YOLO26 detect/segment/pose/
            obb/classify heads (e.g. YOLOE, YOLOWorld or RT-DETR protocols are not validated for RDK export).
    """
    from ultralytics.nn.modules import (
        OBB,
        Classify,
        Depth,
        Detect,
        LRPCHead,
        Pose,
        Pose26,
        RTDETRDecoder,
        Segment,
        Segment26,
        SemanticSegment,
        WorldDetect,
        YOLOEDetect,
    )

    unsupported = (WorldDetect, LRPCHead, YOLOEDetect, RTDETRDecoder, Depth, SemanticSegment)
    description = None
    for module in model.modules():
        if isinstance(module, unsupported):
            raise TypeError(
                f"D-Robotics RDK export supports standard YOLO11/YOLO26 detect, segment, classify, pose and obb "
                f"models, not the {type(module).__name__} protocol."
            )
        forward, task = None, None
        if isinstance(module, Segment26):
            forward, task = _rdk_segment26_forward, "segment"
        elif isinstance(module, Segment):
            forward, task = _rdk_segment_forward, "segment"
        elif isinstance(module, Pose26):
            forward, task = _rdk_pose26_forward, "pose"
        elif isinstance(module, Pose):
            forward, task = _rdk_pose_forward, "pose"
        elif isinstance(module, OBB):  # covers OBB26
            forward, task = _rdk_obb_forward, "obb"
        elif isinstance(module, Classify):
            forward, task = _rdk_classify_forward, "classify"
        elif isinstance(module, Detect):  # covers v10Detect; runs last so task-specific heads match first
            forward, task = _rdk_detect_forward, "detect"
        if forward is not None:
            if task == "classify":  # BPU-friendly 1x1 conv replacing the flatten + Gemm
                linear = module.linear
                module.linear = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=True)
                module.linear.weight.data = linear.weight.data.view(linear.out_features, linear.in_features, 1, 1)
                module.linear.bias.data = linear.bias.data
            module.forward = forward.__get__(module, type(module))
            description = _rdk_metadata(module, task)
            LOGGER.info(
                f"{colorstr('D-Robotics:')} raw {type(module).__name__} head bound "
                f"({description['branch']}, {len(description['outputs'])} outputs)."
            )
    if description is None:
        raise ValueError(
            "D-Robotics RDK export found no supported YOLO head; expected a standard YOLO11/YOLO26 detect, "
            "segment, classify, pose or obb model."
        )
    return description


def _rdk_metadata(head, task):
    """Build the `rdk` schema-1 metadata for one patched head (shapes are filled by the exporter's dry run)."""
    if task == "classify":  # no detect branch, strides or reg_max exist on a Classify head
        return {
            "schema_version": 1,
            "task": task,
            "head_type": type(head).__name__,
            "input_format": "nv12",
            "strides": None,
            "reg_max": None,
            "branch": None,
            "end2end": False,
            "outputs": [{"name": "logits", "role": "logits", "layout": "NHWC"}],
        }
    _rdk_branch(head)  # validates the active branch before any artifact is produced
    strides = [int(s) for s in head.stride.tolist()]
    outputs = []
    for stride in strides:
        outputs.append({"name": f"cls_{stride}", "role": "cls", "layout": "NHWC", "stride": stride})
        outputs.append({"name": f"box_{stride}", "role": "box", "layout": "NHWC", "stride": stride})
        if task == "segment":
            outputs.append({"name": f"mask_{stride}", "role": "mask", "layout": "NHWC", "stride": stride})
        elif task == "pose":
            outputs.append({"name": f"kpt_{stride}", "role": "kpt", "layout": "NHWC", "stride": stride})
        elif task == "obb":
            outputs.append({"name": f"angle_{stride}", "role": "angle", "layout": "NHWC", "stride": stride})
    if task == "segment":
        outputs.append({"name": "proto", "role": "proto", "layout": "NHWC", "stride": 4})  # P3 upsampled 2x
    description = {
        "schema_version": 1,
        "task": task,
        "head_type": type(head).__name__,
        "input_format": "nv12",
        "strides": strides,
        "reg_max": int(getattr(head, "reg_max", 0)) or None,
        "branch": "one2one" if head.end2end else "one2many",
        "end2end": bool(head.end2end),
        # the exporter sets both on the head before patching; the one2one top-k decode must reuse them, not guess
        "max_det": int(head.max_det),
        "agnostic_nms": bool(head.agnostic_nms),
        "outputs": outputs,
    }
    if task == "pose":
        description["kpt_shape"] = [int(k) for k in head.kpt_shape]
    return description


def _write_calibration_files(dataset, cal_dir, prefix=""):
    """Write calibration images as the X5 toolchain's expected RGB float32 NCHW files.

    The loader applies the deployment geometry (detector-like LetterBox at the export size, classification resize +
    center crop) and emits uint8 RGB. The 0-255 float32 bytes match the toolchain's raw NV12 calibration pipeline
    (RGB -> NV12 -> YUV444 -> -128); the compiled model itself applies the configured ``data_scale`` 1/255.
    """
    cal_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in dataset:  # one epoch of the calibration loader; no sampling or count cap
        for image in batch["img"]:
            image.numpy()[None].astype(np.float32).tofile(cal_dir / f"calibration_{count:05d}.rgbchw")
            count += 1
    if not count:
        raise ValueError(f"{prefix} INT8 calibration produced no images from 'data'.")
    LOGGER.info(f"{prefix} wrote {count} calibration images (RGB float32, 0-255) to {cal_dir}")
    return count


def _describe_compiled(bin_file, prefix=""):
    """Describe a compiled .bin through the toolchain's `rdkx5-sim` CLI, returning its schema-1 JSON or None."""
    if shutil.which("rdkx5-sim") is None:
        LOGGER.info(f"{prefix} rdkx5-sim not found on PATH; skipping compiled-spec verification.")
        return None
    result = subprocess.run(
        ["rdkx5-sim", str(bin_file), "--describe"],
        capture_output=True,
        text=True,
        timeout=_DESCRIBE_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{prefix} rdkx5-sim --describe failed ({result.returncode}): {result.stderr.strip()}")
    return {k: v for k, v in json.loads(result.stdout).items() if k in {"inputs", "outputs"}}


def _verify_compiled(spec, outputs, prefix=""):
    """Fail on any compiled-input/output name or layout mismatch with the recorded raw-head contract."""
    inputs = spec["inputs"]
    if len(inputs) != 1 or inputs[0].get("pixel_format") != "nv12":
        raise ValueError(
            f"{prefix} compiled .bin exposes {len(inputs)} input(s) with formats "
            f"{[i.get('pixel_format') for i in inputs]}; expected the single compiled NV12 input."
        )
    compiled = spec["outputs"]
    names = [o["name"] for o in outputs]
    if [o["name"] for o in compiled] != names:
        raise ValueError(
            f"{prefix} compiled .bin outputs {[o['name'] for o in compiled]} do not match the exported ONNX "
            f"outputs {names}; refusing to guess the role binding. Re-export or file an issue with the logs."
        )
    for compiled_out, out in zip(compiled, outputs):
        if list(compiled_out["shape"]) != list(out["shape"]):  # rank may differ only by trailing unit dims
            flat_c, flat_o = int(np.prod(compiled_out["shape"])), int(np.prod(out["shape"]))
            if flat_c != flat_o or list(compiled_out["shape"])[: len(out["shape"])] != list(out["shape"]):
                raise ValueError(
                    f"{prefix} compiled output '{out['name']}' shape {compiled_out['shape']} does not extend the "
                    f"exported shape {out['shape']}."
                )
            LOGGER.warning(
                f"{prefix} compiled output '{out['name']}' deploys as rank-{len(compiled_out['shape'])} "
                f"{compiled_out['shape']} (source graph rank {len(out['shape'])}); recorded in metadata."
            )


def onnx2rdk(onnx_file, output_dir, dataset, metadata, prefix=""):
    """Compile a raw-head ONNX model for the D-Robotics RDK X5 BPU with the external hb_mapper toolchain.

    Args:
        onnx_file (str | Path): Raw-head ONNX model exported by `Exporter.export_rdk`.
        output_dir (str | Path): Final `{model}_rdk_model` directory receiving `{model}.bin` + `metadata.yaml`.
        dataset (InfiniteDataLoader): INT8 calibration loader from `Exporter.get_int8_calibration_dataloader`.
        metadata (dict): Ultralytics export metadata; its `rdk` entry describes the raw outputs.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the final model directory containing the .bin and metadata.yaml.
    """
    import onnx

    assert LINUX and not ARM64, "RDK export is only supported on x86_64 Linux with the hb_mapper toolchain."
    prefix = prefix or colorstr("D-Robotics:")
    onnx_file = Path(onnx_file)
    output_dir = Path(output_dir)
    description = metadata["rdk"]
    graph_names = [o.name for o in onnx.load(str(onnx_file)).graph.output]
    if graph_names != [o["name"] for o in description["outputs"]]:
        raise ValueError(
            f"{prefix} ONNX graph outputs {graph_names} do not match the recorded raw-head outputs "
            f"{[o['name'] for o in description['outputs']]}."
        )
    if shutil.which("hb_mapper") is None:
        raise RuntimeError(
            f"{prefix} 'hb_mapper' not found on PATH. Install the D-Robotics OpenExplore X5 toolchain (Python 3.10) "
            f"and put its bin directory on PATH, e.g. 'PATH=/opt/explore/3.x.x/x86_64/gcc/bin:$PATH'. "
            f"See {TOOLCHAIN_DOC}. Dependencies are never installed automatically."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)  # only this export's own output directory; never a failed artifact mix
    stem = onnx_file.stem
    h, w = metadata["imgsz"]
    model_prefix = f"{stem}_bayese_{h}x{w}_nv12"
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".rdk_work_") as workspace:
        workspace = Path(workspace).resolve()  # absolute: hb_mapper resolves --config against its own cwd
        cal_dir = workspace / "calibration_data"
        bpu_dir = workspace / "bpu_model_output"
        bpu_dir.mkdir(parents=True)
        _write_calibration_files(dataset, cal_dir, prefix)
        config = workspace / "hb_mapper_config.yaml"
        YAML.save(  # safe YAML quoting for paths with spaces or apostrophes
            config,
            {
                "model_parameters": {
                    "onnx_model": str(onnx_file.resolve()),
                    "march": "bayes-e",
                    "layer_out_dump": False,
                    "working_dir": str(bpu_dir.resolve()),
                    "output_model_file_prefix": model_prefix,
                },
                "input_parameters": {
                    "input_name": "",
                    "input_type_rt": "nv12",
                    "input_type_train": "rgb",
                    "input_layout_train": "NCHW",
                    "norm_type": "data_scale",
                    "scale_value": 0.003921568627451,
                },
                "calibration_parameters": {
                    "cal_data_dir": str(cal_dir.resolve()),
                    "cal_data_type": "float32",
                    "calibration_type": "default",
                    "optimization": "set_Softmax_input_int8,set_Softmax_output_int8",
                },
                "compiler_parameters": {
                    "jobs": 16,
                    "compile_mode": "latency",
                    "debug": True,
                    "optimize_level": "O3",
                },
            },
        )
        LOGGER.info(f"{prefix} compiling with hb_mapper makertbin (this can take several minutes)...")
        compile_log = workspace / "hb_mapper_makertbin.log"
        failure_log = output_dir.parent / f"{stem}_hb_mapper_makertbin_failure.log"
        try:
            result = subprocess.run(
                ["hb_mapper", "makertbin", "--config", str(config), "--model-type", "onnx"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_COMPILE_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failure_log.write_text("hb_mapper timed out", encoding="utf-8")
            raise RuntimeError(
                f"{prefix} hb_mapper timed out after {_COMPILE_TIMEOUT}s; log retained at {failure_log}."
            )
        compile_log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            shutil.copy2(compile_log, failure_log)  # keep the toolchain output: the workspace is temporary
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
            raise RuntimeError(
                f"{prefix} hb_mapper failed with exit code {result.returncode}:\n{tail}\n"
                f"Full log retained at {failure_log}."
            )

        compiled_bin = bpu_dir / f"{model_prefix}.bin"
        if not compiled_bin.is_file():
            shutil.copy2(compile_log, failure_log)
            raise FileNotFoundError(
                f"{prefix} hb_mapper finished but '{compiled_bin.name}' was not produced; log retained at "
                f"{failure_log}."
            )
        output_dir.mkdir(parents=True)
        shutil.copy2(compiled_bin, output_dir / f"{stem}.bin")
        quant_info = bpu_dir / f"{model_prefix}_quant_info.json"
        if quant_info.is_file():
            shutil.copy2(quant_info, output_dir)
        shutil.copy2(compile_log, output_dir)

    spec = _describe_compiled(output_dir / f"{stem}.bin", prefix)
    if spec is not None:
        _verify_compiled(spec, description["outputs"], prefix)
        description["compiled"] = spec
    YAML.save(output_dir / "metadata.yaml", metadata)
    LOGGER.info(f"{prefix} export success: {output_dir}")
    return str(output_dir)
