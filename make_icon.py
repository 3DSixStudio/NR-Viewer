# -*- coding: utf-8 -*-
"""Tao icon.ico cho NR Viewer (lồng chữ NR + khối 3D). Chay: python make_icon.py"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size):
    for p in FONTS:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _vgrad(size, top, bot):
    g = Image.new("RGB", (1, size))
    for y in range(size):
        f = y / max(1, size - 1)
        g.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
    return g.resize((size, size))


def render(size):
    S = size * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(S * 0.05)
    rad = int(S * 0.24)

    # nen bo goc + gradient xanh
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], rad, fill=255)
    grad = _vgrad(S, (58, 78, 140), (24, 28, 44)).convert("RGBA")
    img.paste(grad, (0, 0), mask)
    # vien sang nhe
    d.rounded_rectangle([pad, pad, S - pad, S - pad], rad, outline=(120, 150, 220, 255), width=max(2, S // 90))

    # khoi lap phuong iso mo phia sau (goc duoi phai)
    c = np.array([S * 0.70, S * 0.30])
    r = S * 0.20
    cos30, sin30 = math.cos(math.radians(30)), math.sin(math.radians(30))

    def proj(v):
        x, y, z = v
        return c + np.array([(x - z) * cos30, (x + z) * sin30 - y]) * r

    V = {(a, b, cc): proj((a, b, cc)) for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)}
    cube = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cube)
    edges = [((-1, 1, -1), (1, 1, -1)), ((1, 1, -1), (1, 1, 1)), ((1, 1, 1), (-1, 1, 1)), ((-1, 1, 1), (-1, 1, -1)),
             ((-1, 1, 1), (-1, -1, 1)), ((1, 1, 1), (1, -1, 1)), ((1, 1, -1), (1, -1, -1)),
             ((-1, -1, 1), (1, -1, 1)), ((1, -1, 1), (1, -1, -1))]
    lw = max(2, S // 110)
    for a, b in edges:
        cd.line([tuple(V[a]), tuple(V[b])], fill=(120, 215, 255, 255), width=lw)
    for p in [(1, 1, 1)]:
        x, y = V[p]
        rr = S // 55
        cd.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 180, 90, 255))
    cube.putalpha(cube.getchannel("A").point(lambda a: int(a * 0.33)))
    img = Image.alpha_composite(img, cube)
    d = ImageDraw.Draw(img)

    # chu "NR"
    f = _font(int(S * 0.52))
    text = "NR"
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (S - tw) / 2 - bb[0]
    ty = (S - th) / 2 - bb[1] + S * 0.01
    # bong
    d.text((tx + S * 0.012, ty + S * 0.012), text, font=f, fill=(10, 14, 26, 160))
    # chu voi gradient dung
    tmask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(tmask).text((tx, ty), text, font=f, fill=255)
    tgrad = _vgrad(S, (235, 245, 255), (150, 185, 240)).convert("RGBA")
    img.paste(tgrad, (0, 0), tmask)

    return img.resize((size, size), Image.LANCZOS)


def main():
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [render(s) for s in sizes]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    imgs[-1].save(out, format="ICO", sizes=[(s, s) for s in sizes])
    imgs[-1].save(os.path.splitext(out)[0] + ".png")
    print("Da tao", out)


if __name__ == "__main__":
    main()
