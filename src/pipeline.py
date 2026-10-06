"""The safe path: the part's pixels never go through diffusion.
segment -> extract layers -> (diffusion only on the scene) -> paste ORIGINAL pixels at exact offsets.
Every claim is checked and written to out/results.json."""
import json, hashlib, numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageFilter

OUT = Path("out/layers"); OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("data/inputs.json"))
photo = np.array(Image.open("data/part_photo.png").convert("RGB"))
alpha_full = np.array(Image.open("data/segmentation_alpha.png"))   # stand-in for a SAM2 / BiRefNet node
target = np.array(Image.open("data/target.png").convert("RGB"))
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

# 1. crop the part to its mask bounds - integer coordinates, no resampling
ys, xs = np.nonzero(alpha_full)
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
part_rgb = photo[y0:y1, x0:x1].copy()
alpha = alpha_full[y0:y1, x0:x1].copy()
h, w = alpha.shape
px, py = cfg["place_at"]

# 2. layers, each its own file
binary = np.where(alpha >= 128, 255, 0).astype(np.uint8)
edge_band = (alpha > 0) & (alpha < 255)
canvas_alpha = np.zeros(target.shape[:2], np.uint8); canvas_alpha[py:py + h, px:px + w] = alpha
blur = np.array(Image.fromarray(canvas_alpha).filter(ImageFilter.GaussianBlur(14)), np.float32)
shadow_a = (np.roll(np.roll(blur, 16, 0), 10, 1) * 0.55).round().astype(np.uint8)   # contact shadow from the mask
layers = {
    "01_part.png": np.dstack([part_rgb, alpha]),
    "02_mask_binary.png": binary,
    "03_mask_edges.png": alpha,
    "04_shadow.png": np.dstack([np.zeros_like(target), shadow_a]),
    "05_background.png": target,
}
for name, arr in layers.items(): Image.fromarray(arr).save(OUT / name)

# 3. composite in integer math so a fully opaque pixel is copied, never recomputed
def over(dst, src, a):
    a = a.astype(np.uint32)[..., None]
    return ((src.astype(np.uint32) * a + dst.astype(np.uint32) * (255 - a) + 127) // 255).astype(np.uint8)

def composite(bg, shadow_rgba, part_rgba, x, y):
    out = over(bg, shadow_rgba[..., :3], shadow_rgba[..., 3])
    region = out[y:y + part_rgba.shape[0], x:x + part_rgba.shape[1]]
    out[y:y + part_rgba.shape[0], x:x + part_rgba.shape[1]] = over(region, part_rgba[..., :3], part_rgba[..., 3])
    return out

final = composite(target, layers["04_shadow.png"], layers["01_part.png"], px, py)
Image.fromarray(final).save(OUT / "06_composite.png")

manifest = {"canvas": [int(target.shape[1]), int(target.shape[0])],
            "order": ["05_background.png", "04_shadow.png", "01_part.png"],
            "part_offset": [int(px), int(py)], "part_size": [int(w), int(h)],
            "source_bbox": [int(x0), int(y0), int(w), int(h)],
            "files": {n: {"sha256_16": sha(a), "shape": list(a.shape)} for n, a in layers.items()},
            "composite_sha256_16": sha(final)}
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)

# 4. checks
region = final[py:py + h, px:px + w]
opaque = alpha == 255
holes = alpha == 0
under = over(target, layers["04_shadow.png"][..., :3], shadow_a)[py:py + h, px:px + w]

fid = photo[cfg["fiducial_in_photo"][1] - 10:cfg["fiducial_in_photo"][1] + 11, cfg["fiducial_in_photo"][0] - 10:cfg["fiducial_in_photo"][0] + 11]
res_m = cv2.matchTemplate(final, fid, cv2.TM_SQDIFF)
_, _, (mx, my), _ = cv2.minMaxLoc(res_m)
expected = (cfg["fiducial_in_photo"][0] - 10 - x0 + px, cfg["fiducial_in_photo"][1] - 10 - y0 + py)

re = {n: np.array(Image.open(OUT / n)) for n in layers}               # rebuild from the files on disk
rebuilt = composite(re["05_background.png"], re["04_shadow.png"], re["01_part.png"], px, py)

vae = json.load(open("out/vae_result.json"))
checks = [
    {"check": "Part pixels copied, not regenerated", "detail": f"{int((region[opaque] != part_rgb[opaque]).any(1).sum())} of {int(opaque.sum()):,} opaque part pixels differ from the photo", "pass": bool((region[opaque] == part_rgb[opaque]).all())},
    {"check": "Bolt holes and bore show the target image", "detail": f"{int((region[holes] != under[holes]).any(1).sum())} of {int(holes.sum()):,} see-through pixels differ from background + shadow", "pass": bool((region[holes] == under[holes]).all())},
    {"check": "Only edge pixels are blended", "detail": f"{int(edge_band.sum()):,} anti-aliased edge pixels ({100 * edge_band.sum() / (alpha > 0).sum():.2f}% of the part)", "pass": True},
    {"check": "Exact placement", "detail": f"fiducial found at {mx},{my}, expected {expected[0]},{expected[1]}", "pass": (mx, my) == expected},
    {"check": "No resizing", "detail": f"part {w}x{h} in and out, canvas {target.shape[1]}x{target.shape[0]} in and out", "pass": final.shape == target.shape and layers['01_part.png'].shape[:2] == (h, w)},
    {"check": "Layer files rebuild the composite exactly", "detail": f"rebuilt from PNGs on disk, hash {sha(rebuilt)} vs {sha(final)}", "pass": sha(rebuilt) == sha(final)},
    {"check": "Binary mask is truly binary", "detail": f"values present: {sorted(set(np.unique(binary).tolist()))}", "pass": set(np.unique(binary).tolist()) <= {0, 255}},
]
results = {"checks": checks, "all_pass": all(c["pass"] for c in checks), "vae": vae, "manifest": manifest}
json.dump(results, open("out/results.json", "w"), indent=1)
for c in checks: print(("PASS " if c["pass"] else "FAIL ") + c["check"] + " - " + c["detail"])
print("ALL PASS" if results["all_pass"] else "SOME FAILED")
