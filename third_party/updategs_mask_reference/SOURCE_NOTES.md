# UpdateGS mask reference modules

This directory contains a minimal Python source port of 2D mask evidence and proposal code from a remote project named `updateGS`. The copied files came through a supplied handoff bundle; source attribution and file selection follow that bundle's `SOURCE_MANIFEST.json`. The original acquisition chain and the upstream provenance and licenses of the UpdateGS modules remain to be verified.

## Included source

- Eight 2D evidence/proposal modules under `src/actualization_v1/`: `six_method_evidence.py`, `rgb_pair_mask_features.py`, `ssim_evidence.py`, `local_change_evidence.py`, `directional.py`, `semantic.py`, `scar_mask_proposals.py`, and `exact_mask_overlap.py`.
- The required EfficientSAM feature extraction module: `scripts/actualization_v1/extract_render_features.py`.
- Seven EfficientSAM Python modules under `efficient_sam/`: `__init__.py`, `build_efficient_sam.py`, `efficient_sam.py`, `efficient_sam_encoder.py`, `efficient_sam_decoder.py`, `two_way_transformer.py`, and `mlp.py`.
- Empty `__init__.py` package markers in `src/`, `src/actualization_v1/`, `scripts/`, and `scripts/actualization_v1/` make package imports explicit. Imports in the copied reference modules were adjusted to use this directory's package-relative paths.

No additional functional source modules were needed. The upstream cache-builder, protocol guard, model checkpoint, environment dependencies, images, and credentials were not copied.

## EfficientSAM version and license

The text of the seven `efficient_sam/` Python modules listed above matches [yformer/EfficientSAM at commit `d525f622e6f640acf5a0fc37c7ca1f243da5bde0`](https://github.com/yformer/EfficientSAM/tree/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/efficient_sam). Several files, including [the upstream `efficient_sam.py`](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/efficient_sam/efficient_sam.py), carry the copyright notice `Copyright (c) Meta Platforms, Inc. and affiliates.` The upstream commit's [root LICENSE](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/LICENSE) is Apache License 2.0; a copy of that exact file is included as [`efficient_sam/LICENSE`](efficient_sam/LICENSE).

The local files were obtained through the user's remote `updateGS` environment and the handoff bundle described above. Their original acquisition chain is unknown; the text match identifies a corresponding official revision but does not establish that chain. The UpdateGS evidence/proposal modules' provenance and license remain to be verified separately. The upstream [`mlp.py`](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/efficient_sam/mlp.py) references a [MaskFormer source file](https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py); the referenced revision, provenance and applicable terms require separate verification.

EfficientSAM checkpoint weights are supplied separately and were not compared for source or license; no checkpoint is included here. The identified EfficientSAM license does not establish the terms or public distribution status of the rest of this project.

## Importing

Add this directory to `PYTHONPATH` (or `sys.path`) and import, for example:

```python
from src.actualization_v1.six_method_evidence import ...
from src.actualization_v1.scar_mask_proposals import ...
from scripts.actualization_v1.extract_render_features import load_fresh_model
from efficient_sam.efficient_sam import build_efficient_sam
```

The evidence modules require NumPy and SciPy; image feature extraction also uses Pillow, PyTorch, and an EfficientSAM checkpoint supplied separately at runtime. No checkpoint is included here.
