# RDK X5 Modifications Summary

This document summarizes the main source-level changes currently present in `ultralytics_x5` compared with the original workspace copy in `ultralytics`.

## Scope

The changes summarized here are limited to the fork under `ultralytics_x5` and exclude `.git`, cache files, and generated artifacts.

The current differences mainly cover:

- RDK X5 export support
- RDK runtime/backend integration
- Requirement checks and deployment documentation
- A small set of export-related test adjustments

## Core RDK X5 Integration

Files:

- `ultralytics/engine/exporter.py`
- `ultralytics/utils/export/rdk.py`
- `ultralytics/utils/checks.py`
- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

Changes:

- Added a new export format entry: `format="rdk"` with `*_rdk_model/` output directories.
- Added an RDK-specific export flow that:
  - patches model heads before ONNX export
  - exports an intermediate ONNX model with RDK-compatible settings
  - generates an `hb_mapper` configuration for X5 deployment
  - compiles the final `.bin` package and bundles `metadata.yaml`
- Added RDK-specific environment checks for:
  - x86_64 Linux export toolchain availability
  - ARM64 RDK board runtime availability
- Added RDK backend registration to the backend loading path.
- Added a dedicated `RDKBackend` implementation for runtime loading and inference.
- Restricted the current runtime path to X5-only behavior with single-input NV12 inference.
- Fixed circular import issues in the RDK export/runtime path through lazy imports.

## Runtime and Backend Changes

Files:

- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

Changes:

- Moved RDK runtime handling into a dedicated backend module.
- Updated `autobackend.py` to register and dispatch the RDK backend.
- Added X5 runtime preprocessing, runtime loading, and output decoding integration in `RDKBackend`.
- Added stage-level profiling hooks for preprocess, infer, and decode in the RDK backend.
- Added preprocessing optimizations including:
  - direct RGB-to-I420 conversion
  - reusable NV12 scratch buffer allocation
  - batch=1 fast path without `np.stack()`

## ONNX Export Adaptation for RDK

Files:

- `ultralytics/utils/export/rdk.py`

Changes:

- Patched task heads before ONNX export so the exported graph produces raw branch outputs that are easier for RDK conversion.
- Added task-specific export forwards for:
  - detect
  - segment
  - pose
  - obb
  - classify
- For classification export, replaced the linear classifier path with a 1x1 convolution path during export.
- Forced `opset=11` for the intermediate ONNX used by the RDK export flow.

## Requirement and Utility Changes

Files:

- `ultralytics/utils/checks.py`

Changes:

- Added `check_rdk_requirements()`.
- Adjusted RDK requirement checks to distinguish export-side and board-side environments.
- Fixed the `check_requirements()` return behavior for missing packages when `install=False`.

## Documentation Changes

Files:

- `docs/en/guides/rdk-deployment.md`
- `docs/en/integrations/tensorrt.md`
- `docs/en/reference/data/converter.md`
- `docs/overrides/main.html`

Changes:

- Added or updated RDK deployment documentation.
- Updated deployment examples and references related to the RDK flow.
- Included fork-specific documentation adjustments where needed.

## Test Changes

Files:

- `tests/test_exports.py`

Changes:

- Added minimal RDK export-format coverage to the existing export test file.
- Added checks for RDK export format registration and unsupported-platform warning behavior.
- Removed the standalone `tests/test_rdk.py` file and merged its remaining exporter-level coverage into `tests/test_exports.py`.

## Other Source Differences

Files:

- `ultralytics/engine/predictor.py`
- `ultralytics/engine/trainer.py`
- `ultralytics/engine/tuner.py`
- `ultralytics/engine/validator.py`
- `ultralytics/models/yolo/classify/{predict,train,val}.py`
- `ultralytics/models/yolo/detect/{train,val}.py`
- `ultralytics/models/yolo/obb/{predict,train,val}.py`
- `ultralytics/models/yolo/pose/{predict,train,val}.py`
- `ultralytics/models/yolo/segment/{predict,train,val}.py`
- `ultralytics/models/yolo/world/{train,train_world}.py`
- `ultralytics/models/yolo/yoloe/{train,train_seg}.py`
- `ultralytics/models/fastsam/{predict,val}.py`
- `ultralytics/models/sam/predict.py`
- `ultralytics/__init__.py`
- `ultralytics/data/converter.py`
- `ultralytics/hub/auth.py`
- `ultralytics/solutions/solutions.py`

Changes:

- These files differ from the original workspace copy and include fork-specific compatibility changes related to the current development branch.

## 中文说明

本文档用于说明 `ultralytics_x5` 相比工作区中的原始 `ultralytics` 做了哪些源码层面的改动。

### 范围

本文档只描述 `ultralytics_x5` 下的源码差异，不包含 `.git`、缓存文件和生成产物。

当前差异主要包括：

- RDK X5 导出支持
- RDK 运行时与 backend 接入
- 依赖检查与部署文档调整
- 少量导出相关测试调整

### RDK X5 核心集成

涉及文件：

- `ultralytics/engine/exporter.py`
- `ultralytics/utils/export/rdk.py`
- `ultralytics/utils/checks.py`
- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

主要改动：

- 新增 `format="rdk"` 导出格式，输出目录为 `*_rdk_model/`。
- 新增 RDK 专用导出流程，包括：
  - 在 ONNX 导出前对模型 head 做补丁处理
  - 导出中间 ONNX 模型并使用适配 RDK 的设置
  - 生成面向 X5 的 `hb_mapper` 配置
  - 编译最终 `.bin` 包并打包 `metadata.yaml`
- 新增 RDK 环境检查：
  - x86_64 Linux 导出工具链检查
  - ARM64 RDK 板端运行环境检查
- 在 backend 加载路径中新增 RDK backend 注册。
- 新增独立的 `RDKBackend` 用于运行时加载与推理。
- 当前运行时路径收口为 X5-only，使用单输入 NV12 推理。
- 通过延迟导入修复了 RDK 导出与运行时路径中的循环导入问题。

### 运行时与 Backend 改动

涉及文件：

- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

主要改动：

- 将 RDK 运行时处理移动到独立 backend 模块中。
- 更新 `autobackend.py`，使其能够注册并分发 RDK backend。
- 在 `RDKBackend` 中增加 X5 运行时预处理、runtime 加载和输出解码逻辑。
- 为 RDK backend 增加 preprocess / infer / decode 分阶段 profiling。
- 增加预处理优化，包括：
  - 直接 RGB-to-I420 转换
  - 可复用的 NV12 临时 buffer
  - 针对 batch=1 的快路径，避免 `np.stack()`

### RDK 的 ONNX 导出适配

涉及文件：

- `ultralytics/utils/export/rdk.py`

主要改动：

- 在 ONNX 导出前对任务 head 做补丁，使导出的图输出更适合 RDK 转换的原始分支结果。
- 为以下任务增加专用导出 forward：
  - detect
  - segment
  - pose
  - obb
  - classify
- 对分类导出，在导出阶段将线性分类路径替换为 1x1 卷积路径。
- 对 RDK 导出链中的中间 ONNX 强制使用 `opset=11`。

### 依赖检查与工具改动

涉及文件：

- `ultralytics/utils/checks.py`

主要改动：

- 新增 `check_rdk_requirements()`。
- 调整 RDK 依赖检查逻辑，区分导出端与板端运行环境。
- 修复 `check_requirements()` 在 `install=False` 且依赖缺失时的返回行为。

### 文档改动

涉及文件：

- `docs/en/guides/rdk-deployment.md`
- `docs/en/integrations/tensorrt.md`
- `docs/en/reference/data/converter.md`
- `docs/overrides/main.html`

主要改动：

- 新增或更新 RDK 部署文档。
- 更新与 RDK 流程相关的部署示例和引用。
- 按需要加入 fork 专用的文档调整。

### 测试改动

涉及文件：

- `tests/test_exports.py`

主要改动：

- 在现有导出测试文件中增加最小 RDK 导出格式覆盖。
- 增加对 RDK 导出格式注册和不支持平台 warning 行为的检查。
- 删除独立的 `tests/test_rdk.py`，并将其保留的 exporter 级测试合并到 `tests/test_exports.py`。

### 其他源码差异

涉及文件：

- `ultralytics/engine/predictor.py`
- `ultralytics/engine/trainer.py`
- `ultralytics/engine/tuner.py`
- `ultralytics/engine/validator.py`
- `ultralytics/models/yolo/classify/{predict,train,val}.py`
- `ultralytics/models/yolo/detect/{train,val}.py`
- `ultralytics/models/yolo/obb/{predict,train,val}.py`
- `ultralytics/models/yolo/pose/{predict,train,val}.py`
- `ultralytics/models/yolo/segment/{predict,train,val}.py`
- `ultralytics/models/yolo/world/{train,train_world}.py`
- `ultralytics/models/yolo/yoloe/{train,train_seg}.py`
- `ultralytics/models/fastsam/{predict,val}.py`
- `ultralytics/models/sam/predict.py`
- `ultralytics/__init__.py`
- `ultralytics/data/converter.py`
- `ultralytics/hub/auth.py`
- `ultralytics/solutions/solutions.py`

主要改动：

- 这些文件与原始工作区副本存在差异，包含当前开发分支中的 fork 专用兼容性修改。
