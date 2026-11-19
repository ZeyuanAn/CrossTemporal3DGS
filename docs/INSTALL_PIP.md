# Python venv / pip 安装指南

本指南面向 WSL2 Ubuntu 22.04 或 Linux x86_64，使用已有的 Python 3.7.13 解释器。请从仓库根目录执行命令。它是 `environment.yml` / Conda 安装之外的选择。

This guide is for WSL2 Ubuntu 22.04 or Linux x86_64 with an existing Python 3.7.13 interpreter. Run the commands from the repository root. It is an alternative to the Conda setup.

## 验证范围 / What was verified

验证使用 `include-system-site-packages=false` 的 venv 和已有 Python 3.7.13，独立安装 Python 包、重新编译两个扩展，并完成 CUDA 前向/反向检查及 Cup 场景 evidence/pixel 的 BU 3 步、DA 3 步 smoke；使用已有 EfficientSAM checkpoint。

工具链为混合版本：CUDA 11.7.64 `nvcc` 和 11.7 核心组件，另有从 CUDA 11.8 补入的 244 个辅助头文件和库。纯 CUDA 11.7、全新机器及全新 Conda 环境均未验证。以下命令按已验证的依赖版本整理，完整安装序列未在全新环境执行；建议使用完整匹配的 CUDA Toolkit 11.7。

Validation used an isolated venv with an existing Python 3.7.13 interpreter, rebuilt both extensions, and completed CUDA forward/backward checks and a 3+3 iteration Cup evidence/pixel smoke with an existing EfficientSAM checkpoint. The toolkit combined CUDA 11.7.64 compiler/core components with 244 auxiliary files from CUDA 11.8. Pure CUDA 11.7, a fresh machine, and a fresh Conda setup remain unverified. The commands use the validated dependency versions, but the full sequence has not been run in a fresh environment. Use a complete matching CUDA Toolkit 11.7.

## 1. 创建虚拟环境 / Create a venv

使用新目录 `../cross_temporal_venv`；Python 需支持 `venv` 和 `ensurepip`，且不要启用 `--system-site-packages`。

Use a new `../cross_temporal_venv` directory and Python with `venv` and `ensurepip`; leave `--system-site-packages` disabled.

```bash
python3.7 --version
python3.7 -m venv ../cross_temporal_venv
source ../cross_temporal_venv/bin/activate
python -c 'import sys; print(sys.version); assert sys.version_info[:3] == (3, 7, 13)'
```

确认 `../cross_temporal_venv/pyvenv.cfg` 中 `include-system-site-packages = false`。安装兼容 Python 3.7 的 pip、setuptools 和已验证版本的 wheel：

Confirm `include-system-site-packages = false` in `pyvenv.cfg`, then install Python 3.7 compatible packaging tools:

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --upgrade 'pip<24' 'setuptools==67.8.0' 'wheel==0.42.0'
```

## 2. 安装通用依赖 / Install common dependencies

此步骤安装通用依赖及其下游依赖，不安装 torch。

This installs common packages and their dependencies; it does not install torch.

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'numpy==1.21.6' 'scipy==1.7.3' 'Pillow==9.5.0' \
  'plyfile==0.8.1' 'tqdm==4.67.1' 'typing-extensions==4.7.1' \
  'requests==2.31.0' 'opt-einsum==3.3.0'
```

## 3. 安装 CUDA 11.7 版 PyTorch / Install PyTorch for CUDA 11.7

以下两种方式**任选一种**。必须选择 Linux x86_64、CPython 3.7（`cp37-cp37m-linux_x86_64`）且版本带 `+cu117` 的 wheel；不要使用 CPU wheel。`--no-deps` 防止依赖解析更换指定的 torch 版本。

Choose **one** of the following methods. The wheels must be Linux x86_64, CPython 3.7 (`cp37-cp37m-linux_x86_64`), and `+cu117`. `--no-deps` prevents dependency resolution from replacing the selected torch version with a CPU build.

国内网络可使用南大镜像文件直链：

For the Nanjing University mirror, use the direct wheel links:

```bash
python -m pip install --no-deps \
  'https://mirrors.nju.edu.cn/pytorch/whl/cu117/torch-1.13.1%2Bcu117-cp37-cp37m-linux_x86_64.whl' \
  'https://mirrors.nju.edu.cn/pytorch/whl/cu117/torchvision-0.14.1%2Bcu117-cp37-cp37m-linux_x86_64.whl'
```

或使用 PyTorch 官方 cu117 索引：

Or use the official PyTorch cu117 index:

```bash
python -m pip install --no-deps --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cu117 \
  'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117'
```

官方 wheel 列表：[torch](https://download.pytorch.org/whl/cu117/torch/) · [torchvision](https://download.pytorch.org/whl/cu117/torchvision/)。南大直链已检查 HTTP 可访问性；训练短测使用官方 torch wheel 和已有 torchvision wheel，未完整验证这两条镜像直链的安装。

Official wheel lists: [torch](https://download.pytorch.org/whl/cu117/torch/) and [torchvision](https://download.pytorch.org/whl/cu117/torchvision/). The Nanjing links passed an HTTP availability check; the smoke used an official torch wheel and an existing torchvision wheel, so the mirror installation was not fully verified.

用 `--no-deps` 安装剩余依赖，避免重新解析 torch：

Install the remaining packages with `--no-deps` to avoid resolving torch again:

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps \
  'pytorch-msssim==1.0.0' 'lpips==0.1.4'
python -m pip check
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.__version__ == "1.13.1+cu117" and torchvision.__version__ == "0.14.1+cu117" and torch.version.cuda == "11.7"'
```

## 4. 编译扩展 / Build the extensions

准备完整 CUDA Toolkit 11.7，包括 `nvcc`、headers 和库，并确认驱动、GPU 与主机 C/C++ 编译器兼容。修改示例中的 `CUDA_HOME` 指向本机 Toolkit 根目录；`nvcc --version` 应显示 11.7。PyTorch 自带的 CUDA runtime 不能代替编译所需的完整开发工具包。

Install the complete CUDA Toolkit 11.7, including `nvcc`, headers, and libraries, with a compatible driver, GPU, and host compiler. Adjust `CUDA_HOME` to the actual Toolkit root and check that `nvcc --version` reports 11.7. PyTorch's runtime alone is not the development toolkit.

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.7}"
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc
nvcc --version
gcc --version
python -m pip install --no-deps --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-deps --no-build-isolation ./submodules/simple-knn
```

## 5. 检查导入和 GPU / Check imports and GPU

此检查要求可见的 CUDA GPU，并使用 GPU 0；如果需要其他 GPU，可调整 `CUDA_VISIBLE_DEVICES`。检查会导入两个扩展，并在 GPU 上运行一次 `simple-knn`。

This check uses visible GPU 0, imports both extensions, and runs `simple-knn` on CUDA. Adjust `CUDA_VISIBLE_DEVICES` if necessary.

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import torchvision
import lpips
import pytorch_msssim
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('torch CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
assert torch.__version__ == '1.13.1+cu117'
assert torchvision.__version__ == '0.14.1+cu117'
assert torch.version.cuda == '11.7' and torch.cuda.is_available()
points = torch.rand((32, 3), device='cuda', dtype=torch.float32).contiguous()
distances = distCUDA2(points)
assert distances.shape == (32,) and torch.isfinite(distances).all().item()
torch.cuda.synchronize()
print('GPU:', torch.cuda.get_device_name(0), 'simple-knn: OK')
PY
```

训练与数据准备见[中文 README](../README.md)或 [English README](../README.en.md)。`evidence` 模式需要单独提供 EfficientSAM checkpoint；`pixel` 不需要。首次评估时 LPIPS/AlexNet 预训练权重可能联网下载并缓存。

For data preparation and training, see the [Chinese README](../README.md) or [English README](../README.en.md). `evidence` requires a separate EfficientSAM checkpoint; `pixel` does not. LPIPS/AlexNet pretrained weights may download and cache on first evaluation.
