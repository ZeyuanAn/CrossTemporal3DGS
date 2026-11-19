#!/usr/bin/env python3
from __future__ import annotations
"""Fresh same-instance EfficientSAM features for calibrated render/after pairs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

VIEW_IDS = (0, 12, 24, 36)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def image_tensor(path: Path):
    import torch
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None], array.shape[:2]

def load_fresh_model(checkpoint: Path, device: str):
    try:
        import torch
        from efficient_sam.efficient_sam import build_efficient_sam
    except ImportError as exc:
        raise RuntimeError("use the deployed scar3d environment with efficient_sam") from exc
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    return build_efficient_sam(192, 3, checkpoint=str(checkpoint)).to(device).eval(), torch

def fresh_pair_embeddings(model, torch, render_path: Path, after_path: Path):
    render, render_hw = image_tensor(render_path); after, after_hw = image_tensor(after_path)
    if render_hw != after_hw: raise ValueError(f"same-image HW required: {render_hw} vs {after_hw}")
    device = next(model.parameters()).device
    with torch.inference_mode():
        render_embedding = model.get_image_embeddings(render.to(device))[0]
        after_embedding = model.get_image_embeddings(after.to(device))[0]
        self_embedding = model.get_image_embeddings(render.to(device))[0]
    return render_embedding.cpu().numpy(), after_embedding.cpu().numpy(), self_embedding.cpu().numpy(), render_hw

def cosine_difference(render: np.ndarray, after: np.ndarray) -> np.ndarray:
    render = np.asarray(render, np.float32); after = np.asarray(after, np.float32)
    if render.shape != after.shape or render.ndim != 3: raise ValueError("matching CxHf xW embeddings required")
    rn = render / np.maximum(np.linalg.norm(render, axis=0, keepdims=True), 1e-8)
    an = after / np.maximum(np.linalg.norm(after, axis=0, keepdims=True), 1e-8)
    return (1.0 - np.clip(np.sum(rn * an, axis=0), -1.0, 1.0)).astype(np.float32)

def resize_hwc(array: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(array, np.float32), mode="F").resize((hw[1], hw[0]), Image.Resampling.BILINEAR), dtype=np.float32)

def find_render(render_dir: Path, view_id: int) -> Path:
    for candidate in (render_dir / f"view{view_id}" / "history_render_rgb.png", render_dir / f"view_{view_id}" / "history_render_rgb.png", render_dir / f"view{view_id:02d}" / "history_render_rgb.png", render_dir / f"view_{view_id:02d}" / "history_render_rgb.png"):
        if candidate.is_file(): return candidate
    raise FileNotFoundError(f"no history_render_rgb.png for view {view_id} under {render_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", type=Path); ap.add_argument("--after-dir", type=Path)
    ap.add_argument("--render-png", type=Path); ap.add_argument("--after-image", type=Path); ap.add_argument("--view-id", type=int)
    ap.add_argument("--output", type=Path, default=Path("outputs/actualization_v1/20261013/P02_semantic"))
    ap.add_argument("--checkpoint", type=Path, default=Path(".cache/efficientsam/efficient_sam_vitt.pt")); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n1-camera-report", type=Path, default=Path("outputs/actualization_v1/20261013/N1/per_view_pnp.json"))
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.render_png or args.after_image:
        if not (args.render_png and args.after_image and args.view_id in VIEW_IDS): ap.error("single mode requires render-png, after-image, view-id in 0/12/24/36")
        pairs = [(args.view_id, args.render_png, args.after_image)]
    elif args.render_dir and args.after_dir:
        pairs = [(v, find_render(args.render_dir, v), args.after_dir / "frames" / f"{v:06d}.png") for v in VIEW_IDS]
        missing = [str(a) for _, _, a in pairs if not a.is_file()]
        if missing: raise FileNotFoundError("missing observed after RGB: " + ", ".join(missing))
    else: ap.error("provide render-dir/after-dir or explicit single-pair paths")
    model, torch = load_fresh_model(args.checkpoint, args.device)
    camera_report = {}
    if args.n1_camera_report.is_file():
        for record in json.loads(args.n1_camera_report.read_text()):
            camera_report[str(record.get("view"))] = {"pose_c2w": record.get("pose_c2w"), "K_original": record.get("K_original"), "camera_schema": record.get("camera_schema"), "status": record.get("status")}
    manifest = {"method": "fresh_same_instance_efficientsam_render_vs_observed_after", "feature_type": "visualfeature_nonsemantic", "view_ids": [], "cached_after_used": False, "raw_before_index_embedding_used": False, "preprocess": "PIL RGB -> float32 / 255.0 -> BCHW -> EfficientSAM internal ImageNet mean/std", "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "n1_camera_report": str(args.n1_camera_report.resolve()), "pairs": [], "self_compare": {}}
    for view, render, after in pairs:
        hist, obs, self_emb, hw = fresh_pair_embeddings(model, torch, render, after)
        self_diff = cosine_difference(hist, self_emb); diff64 = cosine_difference(hist, obs); diff_hwc = resize_hwc(diff64, hw)
        out = args.output / f"view_{view:02d}.npz"
        np.savez_compressed(out, historical_render_embedding=hist, observed_after_embedding=obs, cosine_difference_64=diff64, feature_difference_hwc=diff_hwc)
        manifest["view_ids"].append(view); manifest["self_compare"][str(view)] = {"cosine_mean": float(1.0 - self_diff.mean()), "difference_max": float(self_diff.max())}
        manifest["pairs"].append({"view_id": view, "historical_render_png": str(render.resolve()), "historical_render_sha256": sha256(render), "observed_after_png": str(after.resolve()), "observed_after_sha256": sha256(after), "image_hw": list(hw), "embedding_shape": list(hist.shape), "output_npz": str(out.resolve()), "n1_camera_final": camera_report.get(str(view))})
    (args.output / "fresh_feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__": main()
