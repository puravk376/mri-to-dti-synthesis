"""
End-to-end architecture figure (IEEE two-column, use with figure*).
Conventional academic style: serif type, square boxes, thin black lines, restrained colour,
real image thumbnails taken from the paper's own Fig. 1 / Fig. 4.

    python make_architecture_figure.py        # needs the thumbnail PNGs in ./thumbs
Edit COUNTS if the dataset changes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

COUNTS = dict(hcp=40, adni=388, total=428, test=43)
THUMBS = "thumbs"
OUT = "architecture_end_to_end"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})
K = "#000000"
GRAY = "#F0F0F0"          # ADNI-only steps
NET = "#DCE6F2"           # learned components (single accent)
NET_E = "#2B4C7E"
MID = "#555555"

W, H = 100.0, 44.0
fig = plt.figure(figsize=(7.16, 7.16 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")


def box(x, y, w, h, fc="white", ec=K, lw=0.55, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def t(x, y, s, size=7, ha="center", va="center", color=K, **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, zorder=6, **kw)


def arr(x0, y0, x1, y1, ls="-", lw=0.55, color=K, conn="arc3", head=0.28):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle=f"-|>,head_length={head},head_width={head * 0.5}",
                                lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0, connectionstyle=conn))


def line(xs, ys, ls="-", lw=0.55, color=K):
    ax.plot(xs, ys, ls=ls, lw=lw, color=color, zorder=3, solid_capstyle="butt")


def thumb(name, x, y, w, h, trim=0.02):
    im = Image.open(os.path.join(THUMBS, name + ".png")).convert("RGB")
    a = np.asarray(im)
    nonwhite = a.min(-1) < 200                       # drop white margins left over from the source figure
    rows = np.where(nonwhite.mean(1) > 0.5)[0]; cols = np.where(nonwhite.mean(0) > 0.5)[0]
    a = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    ty, tx = int(a.shape[0] * trim), int(a.shape[1] * trim)
    a = a[ty:a.shape[0] - ty, tx:a.shape[1] - tx]
    ax.imshow(a, extent=(x, x + w, y, y + h), zorder=5, interpolation="lanczos")
    box(x, y, w, h, fc="none", lw=0.4, z=6)


# ---------------------------------------------------------------- separators + captions
for xs in (24.6, 76.2):
    line([xs, xs], [3.4, 43.4], ls=(0, (3, 2)), lw=0.4, color="#9A9A9A")
t(12.3, 1.5, "(a) Preprocessing and harmonisation", size=7.5)
t(50.4, 1.5, "(b) T1 → tensor synthesis and training objectives", size=7.5)
t(88.2, 1.5, "(c) Evaluation", size=7.5)

# ================================================================ (a)
lw_, sh, gp = 10.9, 2.9, 0.95
lx = (0.9, 12.9)
box(lx[0], 38.2, lw_, 4.0)
t(lx[0] + lw_ / 2, 41.0, r"$\bf{HCP}$", size=7.3)
t(lx[0] + lw_ / 2, 39.25, f"healthy, $n$ = {COUNTS['hcp']}", size=6.2)
box(lx[1], 38.2, lw_, 4.0)
t(lx[1] + lw_ / 2, 41.0, r"$\bf{ADNI}$", size=7.3)
t(lx[1] + lw_ / 2, 39.25, f"CN/MCI/AD, $n$ = {COUNTS['adni']}", size=6.2)

hcp = [("shell subset", r"$b$ = 0, 1000 s/mm$^2$", False),
       ("denoising", "dwidenoise", False),
       ("T1 skull-strip", "HD-BET", False),
       ("tensor fit", "dipy, HCP mask", False)]
adni = [("DICOM → NIfTI", "series ID, dicom2nifti", True),
        ("brain mask", "dwi2mask", True),
        ("denoise + strip", "dwidenoise, HD-BET", False),
        ("tensor fit", "dipy, derived mask", False)]
for x0, steps in zip(lx, (hcp, adni)):
    y = 38.2 - gp - sh
    arr(x0 + lw_ / 2, 38.2, x0 + lw_ / 2, y + sh)
    for i, (a, b, only) in enumerate(steps):
        box(x0, y, lw_, sh, fc=GRAY if only else "white")
        t(x0 + lw_ / 2, y + sh * 0.66, a, size=6.6)
        t(x0 + lw_ / 2, y + sh * 0.28, b, size=5.7, color=MID, style="italic")
        if i < 3:
            arr(x0 + lw_ / 2, y, x0 + lw_ / 2, y - gp)
        y -= sh + gp
y_last = y + sh + gp

hb_y, hb_h = 10.6, 7.2
for x0, dx in zip(lx, (3.0, -3.0)):
    arr(x0 + lw_ / 2, y_last, x0 + lw_ / 2 + dx, hb_y + hb_h)
box(0.9, hb_y, 22.9, hb_h)
t(12.35, hb_y + hb_h - 1.3, "harmonisation to 96 × 112 × 96 grid", size=6.8)
t(12.35, hb_y + 2.55, "world-space (affine) resampling, 2 mm\n"
                      "T1 and DTI share one grid; tensor rotated\n"
                      "T1 percentile norm.; tensor one scale (SPD)", size=5.9, color=MID, linespacing=1.25)
box(0.9, 4.2, 22.9, 4.6)
arr(12.35, hb_y, 12.35, 8.8)
t(12.35, 7.3, f"subject-level split 75/15/10, $N$ = {COUNTS['total']}", size=6.6)
t(12.35, 5.4, f"stratified by group; test set ($n$ = {COUNTS['test']}) fixed", size=5.9, color=MID, style="italic")

# legend for (a)
box(1.0, 42.65, 1.3, 0.85, fc=GRAY, lw=0.4)
t(2.7, 43.07, "ADNI-only step", size=5.7, ha="left", color=MID)
box(13.0, 42.65, 1.3, 0.85, fc=NET, ec=NET_E, lw=0.4)
t(14.7, 43.07, "learned", size=5.7, ha="left", color=MID)

# ================================================================ (b) forward path
yc = 31.4
thumb("t1_canon", 25.6, yc - 3.0, 6.4, 6.0)
t(28.8, yc - 4.2, r"T1 input $X$", size=6.6)

# U-Net as a stack of feature-map blocks
ux = 34.0
spec = [(16, 0), (32, 1), (64, 2), (128, 3), (64, 2), (32, 1), (16, 0)]
blocks = []
x = ux
for c, lv in spec:
    bw = {16: 0.55, 32: 0.8, 64: 1.1, 128: 1.4}[c]
    bh = 7.0 - 1.55 * lv
    y0 = yc + 0.6 - bh / 2 - 0.9 * lv
    box(x, y0, bw, bh, fc=NET, ec=NET_E, lw=0.5)
    blocks.append((x, y0, bw, bh, c))
    x += bw + 0.75
for i, (bx_, by_, bw_, bh_, c) in enumerate(blocks[:4]):
    t(bx_ + bw_ / 2, by_ - 0.75, str(c), size=5.4, color=MID)
for a_, b_ in ((0, 6), (1, 5), (2, 4)):
    xa, ya, wa, ha, _ = blocks[a_]; xb, yb, wb, hb, _ = blocks[b_]
    yy = ya + ha + 0.45 + 0.0 * a_
    line([xa + wa / 2, xa + wa / 2, xb + wb / 2], [ya + ha, yy, yy], lw=0.4, color=NET_E)
    arr(xb + wb / 2, yy, xb + wb / 2, yb + hb, lw=0.4, color=NET_E, head=0.18)
u_end = blocks[-1][0] + blocks[-1][2]
t((ux + u_end) / 2, yc + 6.2, r"3D U-Net $f_\theta$", size=6.8)
arr(32.0, yc, ux, yc)

# heads
hx, hw = u_end + 1.6, 9.0
box(hx, yc + 1.2, hw, 3.6, fc=NET, ec=NET_E)
t(hx + hw / 2, yc + 3.75, "eigenvalue head", size=6.4)
t(hx + hw / 2 + 0.35, yc + 2.05, r"$\vec{\ell}\in\mathbb{R}^3$ (V3, V4: $5\tanh$)", size=5.4)
box(hx, yc - 4.9, hw, 4.6, fc=NET, ec=NET_E)
t(hx + hw / 2, yc - 1.25, "orientation head", size=6.4)
t(hx + hw / 2, yc - 2.75, r"$a_1,a_2\in\mathbb{R}^3$ (6D)", size=5.9)
t(hx + hw / 2, yc - 4.1, r"Gram–Schmidt $\rightarrow R$", size=5.9)
arr(u_end, yc + 0.4, hx, yc + 3.0, conn="arc3,rad=-0.2")
arr(u_end, yc - 0.4, hx, yc - 2.6, conn="arc3,rad=0.2")
t(hx + hw / 2, yc + 5.6, "V1: one 6-channel head", size=5.6, color=MID, style="italic")

# log-tensor, expm, outputs
sx, sw = hx + hw + 1.8, 8.4
box(sx, yc - 1.6, sw, 3.4)
t(sx + sw / 2, yc + 0.1, r"$S=R\,\mathrm{diag}(\vec{\ell}\,)\,R^{T}$", size=6.1)
arr(hx + hw, yc + 3.0, sx, yc + 0.8)
arr(hx + hw, yc - 2.6, sx, yc - 0.8)
box(sx, yc - 7.4, sw, 3.4)
t(sx + sw / 2, yc - 5.7, r"$\mathcal{T}=\exp_m(S)\succ0$", size=6.1)
arr(sx + sw / 2, yc - 1.6, sx + sw / 2, yc - 4.0)
t(sx + sw / 2, yc + 2.5, "log-tensor", size=5.8, color=MID, style="italic")

ox = sx + sw + 1.4
thumb("pred_fa_v4", ox, yc - 0.4, 4.9, 4.9)
thumb("pred_md_v4", ox, yc - 6.6, 4.9, 4.9)
t(ox + 2.45, yc + 5.15, "FA", size=6.3)
t(ox + 2.45, yc - 7.35, "MD", size=6.3)
t(ox + 2.45, yc - 8.6, "(Eq. 2)", size=5.4, color=MID, style="italic")
arr(sx + sw, yc - 5.0, ox, yc + 2.0, conn="arc3,rad=-0.15")
arr(sx + sw, yc - 6.0, ox, yc - 4.2)

# ================================================================ (b) training objectives
ty0 = 12.4
line([25.4, 75.6], [ty0 + 6.2, ty0 + 6.2], ls=(0, (1, 1.5)), lw=0.4, color="#9A9A9A")
t(25.6, ty0 + 5.3, "training only", size=5.8, ha="left", color=MID, style="italic")

# ground truth branch
thumb("gt_fa", 25.6, ty0 - 4.6, 5.0, 5.0)
t(28.1, ty0 - 5.4, r"DTI $\mathcal{T}_{gt}$", size=6.3)
box(32.4, ty0 - 3.6, 7.6, 3.0)
t(36.2, ty0 - 2.1, r"$S_{gt}=\log_m\mathcal{T}_{gt}$", size=6.4)
arr(30.6, ty0 - 2.1, 32.4, ty0 - 2.1)

losses = [
    (41.6, r"$\mathcal{L}_{dir}$", r"$1-|\hat e_1\cdot e_1|$, WM", "V2–V4, ×0.5"),
    (50.1, r"$\mathcal{L}_{recon}$", r"$w\,\Vert S_\theta-S_{gt}\Vert_F^2$", "V1–V4"),
    (58.6, r"$\mathcal{L}_{bg}$", r"$(\mathrm{tr}\,\hat{\mathcal{T}})^2$ off-brain", "V4, ×1"),
    (67.1, r"$\mathcal{L}_{adv}$", r"PatchGAN $D(X,\hat{\mathcal{T}})$", "V3, ×0.005"),
]
lwid = 8.0
for x0, sym, desc, who in losses:
    dashed = sym.endswith("{adv}$")
    bw_ = lwid + (0.8 if dashed else 0)
    box(x0, ty0 - 0.4, bw_, 3.6, ls=(0, (2.5, 1.5)) if dashed else "-")
    t(x0 + bw_ / 2, ty0 + 2.15, sym, size=7)
    t(x0 + bw_ / 2, ty0 + 0.6, desc, size=5.4)
    t(x0 + bw_ / 2, ty0 - 1.2, who, size=5.6, color=MID, style="italic")
# ground-truth bus
bus_y = ty0 - 2.6
line([40.0, 58.6 + lwid / 2], [bus_y, bus_y], lw=0.45, color=MID)
for x0, *_ in losses[:3]:
    arr(x0 + lwid / 2, bus_y, x0 + lwid / 2, ty0 - 0.4 if False else ty0 - 1.85, lw=0.45, color=MID, head=0.2)
t(50.0, bus_y - 0.8, "ground truth", size=5.6, color=MID, style="italic")
# predictions -> losses (dashed)
P = dict(ls=(0, (2, 1.4)), lw=0.45, color=NET_E, head=0.2)
arr(hx + 2.5, yc - 4.9, 41.6 + lwid / 2, ty0 + 3.2, **P)          # R -> L_dir
arr(sx + 1.0, yc - 1.6, 50.1 + lwid / 2, ty0 + 3.2, **P)          # S -> L_recon
arr(sx + sw / 2 - 1.0, yc - 7.4, 58.6 + lwid / 2, ty0 + 3.2, **P)  # T -> L_bg
arr(sx + sw - 1.0, yc - 7.4, 67.1 + lwid / 2, ty0 + 3.2, **P)     # T -> L_adv

t(50.4, 5.0, r"$\mathcal{L}=\mathcal{L}_{recon}+0.5\,\mathcal{L}_{dir}\;[+\,0.005\,\mathcal{L}_{adv}$ in V3$]\;[+\,\mathcal{L}_{bg}$ in V4$]$,"
             r"   $w=\mathrm{FA}_{gt}+0.1$", size=6.4)

# ================================================================ (a) -> (b)
arr(23.8, 16.6, 25.6, yc - 2.0, conn="arc3,rad=-0.25", lw=0.6)
arr(23.8, 12.0, 25.6, ty0 - 2.1, conn="arc3,rad=0.15", lw=0.6)

# ================================================================ (c)
cx0 = 77.2
thumb("gt_fa", cx0 + 0.4, 34.6, 6.6, 6.6)
thumb("pred_fa_v4", cx0 + 13.6, 34.6, 6.6, 6.6)
t(cx0 + 3.7, 33.7, "reference FA", size=6.2)
t(cx0 + 16.9, 33.7, "synthetic FA (V2)", size=6.2)
t(cx0 + 10.3, 37.9, "vs.", size=6.6, style="italic")
arr(ox + 4.9, yc + 2.0, cx0 + 13.6, 36.5, conn="arc3,rad=-0.2", lw=0.6)

box(cx0, 4.2, 21.6, 27.6)
items = [
    ("Voxel-level accuracy", "FA MSE, FA SSIM, MD SSIM"),
    ("Directional agreement", r"$|\hat e_1\cdot e_1|$: all voxels; WM (FA$_{gt}$ > 0.2)"),
    ("Bidirectional synthesis", r"reverse network $\hat{\mathcal{T}}\rightarrow$ T1, SSIM"),
    ("Diagnostic classification", "random forest, 7 FA/MD features,\nAD / MCI / CN, real vs. synthetic"),
    ("Tractography overlap", r"deterministic tracking along $e_1$," "\nDice of streamline masks"),
]
yy = 29.6
for i, (h_, d_) in enumerate(items):
    t(cx0 + 1.0, yy, f"{i + 1}. {h_}", size=6.6, ha="left", weight="bold")
    t(cx0 + 2.1, yy - 1.75 - (0.6 if "\n" in d_ else 0), d_, size=6.0, ha="left", color="#222222", linespacing=1.2)
    yy -= 5.2 if "\n" in d_ else 4.4
    if i < len(items) - 1:
        line([cx0 + 1.0, cx0 + 20.6], [yy + 1.6, yy + 1.6], lw=0.3, color="#BBBBBB")
t(cx0 + 10.8, 6.0, f"held-out test set, $n$ = {COUNTS['test']}", size=5.9, color=MID, style="italic")

fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT)
