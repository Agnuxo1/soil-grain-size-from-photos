"""Render scientific GIFs from an auditable data/diagram specification.

SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
SPDX-License-Identifier: MIT
Research-Project: soil-grain-size-from-photos
Research-Date: 2026-11-04
Research-ORCID: https://orcid.org/0009-0001-1634-7063
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

W,H=1200,470
PAPER="#f7f9fb"
INK="#142337"
GRID="#d9e3ec"
MUTED="#43596c"
TEAL="#087f86"
BLUE="#284f90"
GOLD="#a85b12"
PURPLE="#715285"
LINES=(TEAL,BLUE,GOLD,PURPLE)
ROOT=Path(__file__).resolve().parents[1]

@lru_cache
def font(size, bold=False):
    names=("segoeuib.ttf","DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf","DejaVuSans.ttf")
    for n in names:
        for loc in (n,str(Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts"/n)):
            try:
                return ImageFont.truetype(loc,size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)

def label(d,xy,s,size,color=INK,bold=False,width=None,anchor=None):
    f=font(size,bold)
    while width and f.getlength(s)>width and size>13:
        size-=1
        f=font(size,bold)
    d.text(xy,s,font=f,fill=color,anchor=anchor)

def lines(s):
    return s.split("\n")

def page(spec,n,fig):
    im=Image.new("RGB",(W,H),PAPER)
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,7),fill=TEAL)
    label(d,(42,20),f"FIG. {n:02d}   |   {fig['basis'].upper()}",17,TEAL,True)
    label(d,(41,49),fig["title"],36,INK,True,1117)
    label(d,(42,97),fig["subtitle"],20,MUTED,False,1115)
    d.line((42,406,1156,406),fill=GRID,width=2)
    label(d,(42,419),fig["source"],17,MUTED,False,1000)
    label(d,(42,444),f"{spec['title']}  ·  Francisco Angulo de Lafuente  ·  ORCID 0009-0001-1634-7063",15,INK,False,1110)
    return im,d

def cap(d,x,y,a,c,w=3):
    d.line((x,y,x-11*math.cos(a-.52),y-11*math.sin(a-.52)),fill=c,width=w)
    d.line((x,y,x-11*math.cos(a+.52),y-11*math.sin(a+.52)),fill=c,width=w)

def flow(spec,n,f):
    ns=f["nodes"]
    coords=[(50+v[0]*1100,155+v[1]*210) for v in ns]
    edges=f["edges"]
    for t in (1,0,.13,.26,.39,.52,.65,.78,.90,1):
        im,d=page(spec,n,f)
        for j,(a,b,*rest) in enumerate(edges):
            x,y=coords[a]; u,v=coords[b]
            angle=math.atan2(v-y,u-x)
            dx,dy=math.cos(angle),math.sin(angle)
            start=(x+18*dx,y+18*dy); end=(u-19*dx,v-19*dy)
            d.line([start,end],fill=GRID,width=5)
            cap(d,*end,angle,GRID,4)
            progress=max(0,min(1,t*1.38-j*.11))
            if progress>0:
                p=(start[0]+progress*(end[0]-start[0]),start[1]+progress*(end[1]-start[1]))
                d.line([start,p],fill=LINES[j%len(LINES)],width=4)
                cap(d,*p,angle,LINES[j%len(LINES)],3)
            if rest and rest[0]:
                mx=(x+u)/2; my=(y+v)/2-14
                st=rest[0]; ff=font(15)
                wid=ff.getlength(st)+10
                d.rectangle((mx-wid/2,my-9,mx+wid/2,my+9),fill=PAPER)
                label(d,(mx,my),st,15,MUTED,anchor="mm")
        for i,((x,y),v) in enumerate(zip(coords,ns)):
            d.ellipse((x-16,y-16,x+16,y+16),fill=PAPER,outline=TEAL,width=3)
            label(d,(x,y),str(i+1),16,TEAL,True,anchor="mm")
            anchor=v[3] if len(v)>3 else "below"
            for j,txt in enumerate(lines(v[2])):
                off=25+j*22 if anchor=="below" else -40-(len(lines(v[2]))-1-j)*22
                label(d,(x,y+off),txt,20,INK,True,255,anchor="mm")
        yield im

def bars(spec,n,f):
    names=f["labels"]; vals=f["values"]; errors=f.get("intervals",[None]*len(vals))
    mn,mx=f["lim"]; ticks=f.get("ticks",[mn+(mx-mn)*i/5 for i in range(6)])
    x0,x1,y0,y1=325,1074,153,350
    scale=lambda a:x0+(x1-x0)*(a-mn)/(mx-mn)
    zero=scale(max(mn,min(0,mx)))
    for t in (1,.05,.16,.32,.50,.68,.84,1):
        im,d=page(spec,n,f)
        for tick in ticks:
            xx=scale(tick)
            d.line((xx,y0,xx,y1),fill=GRID,width=2)
            label(d,(xx,y1+9),f"{tick:g}",18,MUTED,anchor="mt")
        d.line((x0,y1,x1,y1),fill=INK,width=2)
        d.line((zero,y0,zero,y1),fill=INK,width=2)
        label(d,((x0+x1)/2,389),f["xlabel"],17,MUTED,True,anchor="mm")
        space=(y1-y0)/len(vals)
        for j,(v,name,ci) in enumerate(zip(vals,names,errors)):
            y=y0+(j+.52)*space
            end=zero+(scale(v)-zero)*t
            c=LINES[j%len(LINES)]
            label(d,(x0-15,y),name,21,INK,False,275,anchor="rm")
            d.rectangle((min(zero,end),y-11,max(zero,end),y+11),fill=c)
            if ci and t==1:
                lo,hi=map(scale,ci)
                d.line((lo,y,hi,y),fill=INK,width=3)
                d.line((lo,y-7,lo,y+7),fill=INK,width=2)
                d.line((hi,y-7,hi,y+7),fill=INK,width=2)
            if t==1:
                out=f["formats"][j] if f.get("formats") else f"{v:g}"
                place=scale(ci[1])+10 if ci else (end+9 if v>=0 else end-9)
                label(d,(place,y),out,18,c,True,anchor="lm" if v>=0 or ci else "rm")
        yield im

def xy(spec,n,f):
    series=f["series"]
    xmin,xmax=f["xlim"]; ymin,ymax=f["ylim"]
    l,r,t,b=122,1054,164,344
    mapx=lambda x:l+(r-l)*(x-xmin)/(xmax-xmin)
    mapy=lambda y:b-(b-t)*(y-ymin)/(ymax-ymin)
    for frac in (1,.08,.19,.32,.44,.56,.68,.81,.93,1):
        im,d=page(spec,n,f)
        for tick in f.get("xticks",[xmin+(xmax-xmin)*i/5 for i in range(6)]):
            x=mapx(tick)
            d.line((x,t,x,b),fill=GRID,width=2)
            label(d,(x,b+10),f"{tick:g}",17,MUTED,anchor="mt")
        for tick in f.get("yticks",[ymin+(ymax-ymin)*i/4 for i in range(5)]):
            y=mapy(tick)
            d.line((l,y,r,y),fill=GRID,width=2)
            label(d,(l-13,y),f"{tick:g}",17,MUTED,anchor="rm")
        d.line((l,t,l,b,r,b),fill=INK,width=2)
        label(d,((l+r)/2,392),f["xlabel"],18,INK,True,anchor="mm")
        label(d,(l,143),f["ylabel"],18,INK,True,anchor="lm")
        for i,s in enumerate(series):
            c=LINES[i%len(LINES)]
            pts=[(mapx(a),mapy(v)) for a,v in s["points"]]
            if len(pts)>1:
                d.line(pts,fill=GRID,width=2)
            k=max(1,math.ceil(frac*len(pts)))
            if k>1:
                d.line(pts[:k],fill=c,width=4)
            for x,y in (pts[:k] if len(pts)<12 else [pts[k-1]]):
                d.ellipse((x-4,y-4,x+4,y+4),fill=c)
            xk=325+i*229
            d.line((xk,143,xk+23,143),fill=c,width=4)
            label(d,(xk+30,143),s["label"],16,INK,False,191,anchor="lm")
        yield im

def matrix(spec,n,f):
    values=f["values"]
    nx=len(values[0]); ny=len(values)
    x0,x1,y0,y1=237,954,151,352
    dx=(x1-x0)/nx; dy=(y1-y0)/ny
    mn,mx=f.get("lim",[min(map(min,values)),max(map(max,values))])
    def rgb(v):
        q=max(0,min(1,(v-mn)/(mx-mn or 1)))
        return tuple(round((1-q)*a+q*b) for a,b in zip((235,245,247),(1,85,106)))
    for fr in (1,.12,.25,.38,.51,.64,.77,.89,1):
        im,d=page(spec,n,f)
        for j,row in enumerate(values):
            for i,value in enumerate(row):
                col=rgb(value) if i < math.ceil(nx*fr) else (228,237,240)
                bounds=(x0+i*dx,y0+j*dy,x0+(i+1)*dx-2,y0+(j+1)*dy-2)
                d.rectangle(bounds,fill=col)
                if f.get("annotate") and i < math.ceil(nx*fr):
                    label(d,((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2),f"{value:g}",18,
                          INK if (value-mn)/(mx-mn or 1)<.45 else "#ffffff",True,anchor="mm")
            if f.get("rows"):
                label(d,(x0-14,y0+(j+.5)*dy),f["rows"][j],18,MUTED,False,181,anchor="rm")
        for i,v in enumerate(f.get("columns",[])):
            label(d,(x0+(i+.5)*dx,y1+12),str(v),17,MUTED,False,int(dx)-5,anchor="mt")
        for y in range(y0,y1):
            v=mx-(mx-mn)*(y-y0)/(y1-y0)
            d.line((1018,y,1042,y),fill=rgb(v),width=1)
        label(d,(1050,y0),f"{mx:g}",17,INK,anchor="lm")
        label(d,(1050,y1),f"{mn:g}",17,INK,anchor="lm")
        label(d,((x0+x1)/2,392),f["xlabel"],17,INK,True,anchor="mm")
        yield im

def banner(spec,path):
    width,height=1200,355
    art=ImageOps.fit(Image.open(path).convert("RGB"),(width,height))
    backdrop=Image.new("RGB",(width,height),"#0b1726")
    mask=Image.new("L",(width,height))
    mask.putdata([round(254*max(0,min(1,(827-x)/435))) for x in range(width)]*height)
    out=Image.composite(backdrop,art,mask)
    for t in range(14):
        im=out.copy();d=ImageDraw.Draw(im)
        d.line((55,52,510,52),fill="#48627b",width=2)
        xs=range(55,511,4)
        pts=[(x,52+12*math.sin((x-55)*.028)*math.exp(-((x-282)/137)**2)) for x in xs]
        d.line(pts[:max(2,int(len(pts)*(t+1)/14))],fill="#69e1d8",width=3)
        label(d,(54,75),"RESEARCH SOFTWARE  /  COMPETITION EDITION",18,"#79ddd5",True)
        label(d,(51,125),spec["title"],61,"#f7f9fb",True,685)
        label(d,(55,208),spec["subtitle"],25,"#d6e7f0",False,600)
        label(d,(55,302),"FRANCISCO ANGULO DE LAFUENTE  ·  ORCID 0009-0001-1634-7063",16,"#c9dce7",True)
        yield im

def save(path,frames,spec,is_banner=False):
    imgs=list(frames)
    palette=imgs[0].quantize(colors=184,method=Image.Quantize.MEDIANCUT)
    frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in imgs]
    delay=[110]*len(frames) if is_banner else [1050]+[410]*(len(frames)-2)+[1800]
    if is_banner: delay[-1]=1600
    txt=f"Project: {spec['title']}; Date: 2026-11-04; Author: Francisco Angulo de Lafuente; ORCID: 0009-0001-1634-7063"
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=delay,loop=0,optimize=True,
                   disposal=1,comment=txt.encode("utf-8"))
    imgs[-1].save(path.with_suffix(".png"),optimize=True)

def main():
    d=ROOT/"docs"/"figures"
    spec=json.loads((d/"figures.json").read_text(encoding="utf-8"))
    if (d/"cover.jpg").exists():
        save(d/"cover.gif",banner(spec,d/"cover.jpg"),spec,True)
    funcs={"flow":flow,"bars":bars,"xy":xy,"matrix":matrix}
    for n,f in enumerate(spec["figures"],1):
        save(d/f"{f['id']}.gif",funcs[f["kind"]](spec,n,f),spec)
if __name__=="__main__":
    main()
