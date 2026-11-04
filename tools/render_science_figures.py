"""Render deterministic science diagrams for the repository.

SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
SPDX-License-Identifier: MIT
Research-Project: soil-grain-size-from-photos
Research-Date: 2026-11-04
Research-ORCID: https://orcid.org/0009-0001-1634-7063
"""
import argparse
from pathlib import Path
import json
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

W = 1100
NAVY, PANEL, INK, MUTED, TEAL, GOLD = "#0a1526", "#122137", "#f0f6fa", "#bdccd9", "#62dbd1", "#f3b965"

def font(size, bold=False):
    names = ["segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "DejaVuSans.ttf"]
    for name in names:
        for file in (name, str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name)):
            try:
                return ImageFont.truetype(file, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)

def label(draw, point, value, size, color, bold=False, max_width=None):
    ft = font(size, bold)
    if max_width:
        while ft.getlength(value) > max_width and size > 13:
            size -= 1
            ft = font(size, bold)
    draw.text(point, value, font=ft, fill=color)

def base(height):
    img = Image.new("RGB", (W, height), NAVY)
    draw = ImageDraw.Draw(img)
    for x in range(26, W, 31):
        for y in range(17, height, 31):
            draw.point((x, y), fill="#243b50")
    return img, draw

def compose_banner(spec, path):
    H = 337
    with Image.open(path) as im:
        art = ImageOps.fit(im.convert("RGB"), (W,H), centering=(0.5,0.5))
    shade = Image.new("RGB",(W,H),NAVY)
    mask = Image.new("L",(W,H))
    mask.putdata([int(249 * max(0, min(1, (800 - x) / 330))) for x in range(W)] * H)
    background = Image.composite(shade, art, mask)
    for i in range(18):
        img = background.copy()
        dr = ImageDraw.Draw(img)
        x = round(57+int(330*i/17))
        dr.line((57, 54, 387, 54), fill="#395268", width=2)
        dr.line((57, 54, x, 54), fill=TEAL, width=3)
        label(dr,(57,68),"COMPETITION RESEARCH  /  2026",18,TEAL,True)
        label(dr,(54,117),spec["title"],57,INK,True,600)
        label(dr,(57,200),spec["sub"],24,MUTED,max_width=540)
        label(dr,(57,247),spec["issue"],17,GOLD,max_width=500)
        label(dr,(57,302),"FRANCISCO ANGULO DE LAFUENTE  /  SCIENTIFIC SOFTWARE",13,MUTED,True)
        yield img

def diagram(project, title, steps, notes, idx, observed=False):
    for active in [None, *range(len(steps)), None]:
        img, dr = base(264)
        label(dr,(30,16),f"FIGURE {idx:02d}  /  {'REPORTED EVIDENCE' if observed else 'METHOD SCHEMATIC'}",15,TEAL,True)
        label(dr,(29,40),title,34,INK,True,1025)
        left, gap = 30, 29
        width = (W - 2 * left - (len(steps) - 1) * gap) / len(steps)
        for i,(step,note) in enumerate(zip(steps,notes)):
            x = int(left+i*(width+gap)); right=int(x+width)
            highlight = active is None or i == active
            edge = TEAL if i == active else "#37556d"
            dr.rounded_rectangle((x,108,right,204),radius=12,fill="#153647" if i==active else PANEL,outline=edge,width=2)
            label(dr,(x+12,118),f"{i+1:02d}",14,TEAL if highlight else MUTED,True)
            label(dr,(x+12,142),step,24,INK,True,width-22)
            label(dr,(x+12,174),note,17,MUTED,False,width-22)
            if i < len(steps)-1:
                end=right+gap-5
                dr.line((right+4,157,end,157),fill=edge,width=2)
                dr.line((end-4,153,end,157,end-4,161),fill=edge,width=2)
        dr.line((30,228,1070,228),fill="#2c4257",width=1)
        label(dr,(30,236),project.upper(),13,MUTED,True)
        label(dr,(810,236),"ORCID 0009-0001-1634-7063",13,MUTED)
        yield img

def save(path, frames, project, banner=False):
    frames=list(frames)
    palette=frames[0].quantize(colors=148,method=Image.Quantize.MEDIANCUT)
    frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    dwell = [115] * len(frames) if banner else [1000] + [660]*(len(frames)-2) + [1150]
    if banner:
        dwell[-1]=1550
    note = f"Project: {project}; Date: 2026-11-04; Author: Francisco Angulo de Lafuente; ORCID: 0009-0001-1634-7063"
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=dwell,loop=0,
                   optimize=True,disposal=1,comment=note.encode("utf-8"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    out=args.root/"docs"/"science-figures"
    spec=json.loads((out/"figure-spec.json").read_text(encoding="utf-8"))
    save(out/"header.gif",compose_banner(spec, args.root/spec["header_source"]),spec["title"],True)
    for idx,(_,title,steps,notes) in enumerate(spec["sections"],1):
        save(out/f"{idx:02d}.gif",diagram(spec["title"],title,steps,notes,idx,
              observed=title.startswith("Measure OOF")),spec["title"])

if __name__=="__main__":
    main()
