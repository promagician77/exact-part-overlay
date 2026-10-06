"""Generates the test inputs: a machined flange photographed on a studio sweep,
and a separate target image (an assembly fixture) to overlay it onto.
Stand-ins for the client's real photos - same hard details: fine text, edges, holes."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json, math

rng = np.random.default_rng(7)
W, H = 1200, 900
CX, CY, R = 600, 450, 300            # flange center and radius
SS = 4                               # supersampling for the mask

def studio_sweep(w, h):
    y = np.linspace(0, 1, h)[:, None]; x = np.linspace(-1, 1, w)[None, :]
    base = 222 - 38 * y - 14 * x**2
    img = np.repeat(base[..., None], 3, 2) + rng.normal(0, 1.6, (h, w, 3))
    img[..., 2] += 4; img[..., 0] -= 2
    return img

def geometry_mask(scale):
    """Exact part outline: disc, minus central bore and six bolt holes."""
    w, h, s = W * scale, H * scale, scale
    m = Image.new("L", (w, h), 0); d = ImageDraw.Draw(m)
    d.ellipse([(CX - R) * s, (CY - R) * s, (CX + R) * s, (CY + R) * s], fill=255)
    d.ellipse([(CX - 78) * s, (CY - 78) * s, (CX + 78) * s, (CY + 78) * s], fill=0)
    for k in range(6):
        a = math.radians(30 + 60 * k); hx, hy = CX + 205 * math.cos(a), CY + 205 * math.sin(a)
        d.ellipse([(hx - 24) * s, (hy - 24) * s, (hx + 24) * s, (hy + 24) * s], fill=0)
    return m

def render():
    bg = studio_sweep(W, H)
    soft = np.array(geometry_mask(SS).resize((W, H), Image.BOX)).astype(np.float32) / 255.0

    sh = Image.fromarray((soft * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))
    sh = np.roll(np.roll(np.array(sh, np.float32) / 255, 22, 0), 14, 1)
    bg *= (1 - 0.42 * sh)[..., None]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - CX, yy - CY; r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    metal = 168 + 34 * np.cos(2 * (th - 0.6)) - 0.05 * (r - 160)
    metal += 6 * np.sin(r * 2.1) + 3 * np.sin(r * 5.3 + 1.1)
    metal += rng.normal(0, 2.2, (H, W))
    knurl = (r > 250) & (r < 266)
    metal[knurl] += 22 * np.sign(np.sin(th[knurl] * 360))
    chamfer = (r > 286) & (r < 300)
    metal[chamfer] += 40 * np.clip(np.cos(th[chamfer] - 2.3), 0, 1) - 10
    bore_ch = (r > 78) & (r < 92)
    metal[bore_ch] -= 30 * np.clip(np.cos(th[bore_ch] + 0.8), 0, 1) + 6
    part = np.stack([metal * 0.97, metal * 0.99, metal * 1.03], -1)

    img = bg * (1 - soft[..., None]) + part * soft[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    f_sn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    f_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    d.text((CX - 60, CY + 112), "SN 40871-B", font=f_sn, fill=(58, 60, 64))
    d.text((CX - 44, CY + 133), "Ø52 H7  6061-T6", font=f_sm, fill=(70, 72, 76))
    d.text((CX - 36, CY - 140), "REV C  12/26", font=f_sm, fill=(70, 72, 76))
    for t in range(-6, 7):
        d.point((CX + 150 + t, CY - 40), fill=(40, 40, 44)); d.point((CX + 150, CY - 40 + t), fill=(40, 40, 44))
    return np.array(pil), soft

def target_image():
    TW, TH = 1400, 1000
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    plate = 74 + 10 * np.sin(xx / 3.1) * np.cos(yy / 47) + rng.normal(0, 3, (TH, TW))
    img = np.stack([plate * 0.94, plate * 1.0, plate * 1.08], -1)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(pil)
    for gx in range(50, TW, 100): d.line([(gx, 0), (gx, TH)], fill=(92, 98, 108), width=1)
    for gy in range(50, TH, 100): d.line([(0, gy), (TW, gy)], fill=(92, 98, 108), width=1)
    for (px, py) in [(120, 120), (1280, 120), (120, 880), (1280, 880)]:
        d.ellipse([px - 18, py - 18, px + 18, py + 18], fill=(30, 32, 36), outline=(120, 126, 136), width=3)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    d.text((40, TH - 40), "FIXTURE F-12  STATION 3", font=f, fill=(150, 156, 166))
    return np.array(pil)

if __name__ == "__main__":
    src, soft = render()
    Image.fromarray(src).save("data/part_photo.png")
    Image.fromarray((soft * 255).round().astype(np.uint8)).save("data/segmentation_alpha.png")
    Image.fromarray(target_image()).save("data/target.png")
    json.dump({"part_bbox": [CX - R, CY - R, 2 * R, 2 * R], "fiducial_in_photo": [CX + 150, CY - 40],
               "place_at": [380, 190]}, open("data/inputs.json", "w"))
    print("inputs written")
