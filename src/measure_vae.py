"""Measures what one Stable Diffusion latent round-trip does to the part - no prompt, no sampling."""
import sys, json, time, numpy as np
from PIL import Image
sys.path.insert(0, "src")
from taesd_onnx import TAESD

CROP = (380, 400, 892, 784)          # serial number, bolt hole, knurl ring, chamfer
photo = np.array(Image.open("data/part_photo.png").convert("RGB"))
alpha = np.array(Image.open("data/segmentation_alpha.png"))
crop = photo[CROP[1]:CROP[3], CROP[0]:CROP[2]]
a = alpha[CROP[1]:CROP[3], CROP[0]:CROP[2]] == 255

t = time.time()
model = TAESD("data/taesd_encoder.pth", "data/taesd_decoder.pth")
out, lat = model.roundtrip(crop)
secs = time.time() - t

diff = np.abs(out.astype(int) - crop.astype(int))
d = diff.max(2)
mse = (diff.astype(float) ** 2).mean()
psnr = 10 * np.log10(255 ** 2 / mse)
part = d[a]
res = {
    "model": "TAESD (official weights, github.com/madebyollin/taesd)",
    "crop": CROP, "pixels": int(d.size), "part_pixels": int(a.sum()),
    "latent_shape": list(lat.shape),
    "part_changed_pct": round(100 * (part > 0).mean(), 2),
    "part_changed_over_8_pct": round(100 * (part > 8).mean(), 2),
    "part_changed_over_24_pct": round(100 * (part > 24).mean(), 2),
    "part_mean_abs_err": round(float(diff[a].mean()), 2),
    "max_err": int(d.max()), "psnr_db": round(float(psnr), 2), "seconds": round(secs, 1),
}
Image.fromarray(crop).save("out/vae_before.png")
Image.fromarray(out).save("out/vae_after.png")
heat = np.clip(d * 6, 0, 255).astype(np.uint8)
hm = np.stack([heat, (heat * 0.35).astype(np.uint8), np.zeros_like(heat)], -1)
Image.fromarray(hm).save("out/vae_diff.png")
json.dump(res, open("out/vae_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
