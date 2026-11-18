"""Generate the demo GIFs of the README from the implementation in ex2.py (headless, no windows).

    python docs/make_gifs.py            writes docs/*.gif

Every frame is computed with the functions of ex2.py: the warps and morphs with warp_lines / morph, the 3D
views with the viewer's drawing code on top of rotate, quaternions, slerp, look_at, perspective, viewport,
project and chase_view.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import ex2      # noqa: E402
import viewer   # noqa: E402


def grab(fig):
    """A matplotlib figure as a PIL image."""
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return im


def save_gif(frames, name, ms=45, colors=112, hold=14):
    """Write the frames as a looping GIF, holding the first and last frame for a moment."""
    frames = [frames[0]] * hold + list(frames) + [frames[-1]] * hold
    pal = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    path = os.path.join(HERE, name)
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True, disposal=2)
    print(f"{name}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.2f} MB")


def boomerang(frames):
    return list(frames) + list(frames[-2:0:-1])


# ---------------------------------------------------------------- 1. Beier-Neely warp
def gif_warp():
    img = imread(os.path.join(ROOT, "test_image.png"))[..., :3].astype(float)
    src = np.array([[[64, 128], [192, 128]]], float)
    # destination line: translated, rotated by 90 degrees about its midpoint, doubled in length, then back
    key = [[(64, 128), (192, 128)], [(84, 148), (212, 148)], [(128, 64), (128, 192)], [(0, 128), (256, 128)],
           [(64, 128), (192, 128)]]
    frames = []
    for a, b in zip(key[:-1], key[1:]):
        for t in np.linspace(0, 1, 16, endpoint=False):
            P = (1 - t) * np.array(a[0], float) + t * np.array(b[0], float)
            Q = (1 - t) * np.array(a[1], float) + t * np.array(b[1], float)
            dst = np.array([[P, Q]])
            out = ex2.warp_lines(img, src, dst, 1, 2, 0.5)
            fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.5), dpi=100)
            for a_, im, L, title in ((ax[0], img, src[0], "source image, line $P'Q'$"),
                                     (ax[1], out, dst[0], "output (Beier–Neely), line $PQ$")):
                a_.imshow(np.clip(im, 0, 1), interpolation="nearest")
                a_.annotate("", xy=L[1], xytext=L[0], arrowprops=dict(arrowstyle="-|>", color="#e00000", lw=2.2))
                a_.set_xlim(-0.5, 255.5)
                a_.set_ylim(255.5, -0.5)
                a_.set_title(title, fontsize=10)
                a_.axis("off")
            fig.tight_layout(pad=0.6)
            frames.append(grab(fig))
    save_gif(frames, "warp.gif", ms=55)


# ---------------------------------------------------------------- 2. face morph
LINES_A = [((205, 345), (252, 342)), ((372, 342), (418, 346)), ((160, 322), (262, 308)), ((372, 308), (470, 322)),
           ((318, 380), (318, 470)), ((268, 468), (365, 468)), ((240, 535), (388, 535)), ((125, 420), (235, 640)),
           ((235, 655), (395, 655)), ((400, 640), (495, 420)), ((90, 330), (310, 25)), ((310, 25), (535, 330))]
LINES_B = [((190, 357), (262, 357)), ((362, 357), (445, 355)), ((160, 300), (265, 295)), ((358, 295), (470, 302)),
           ((318, 390), (318, 470)), ((258, 468), (372, 468)), ((242, 545), (388, 545)), ((135, 420), (225, 620)),
           ((225, 640), (400, 640)), ((405, 620), (495, 420)), ((80, 270), (310, 15)), ((310, 15), (555, 270))]


def gif_morph():
    scale = 0.5
    imgs = []
    for f in ("morph_a.jpg", "morph_b.jpg"):
        im = Image.open(os.path.join(ROOT, f)).convert("RGB")
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        imgs.append(np.asarray(im, float) / 255)
    L0 = np.array(LINES_A, float) * scale
    L1 = np.array(LINES_B, float) * scale
    frames = []
    for t in np.linspace(0, 1, 28):
        out = ex2.morph(imgs[0], imgs[1], L0, L1, t, 1, 2, 0.5)
        im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(im).text((8, 6), f"t = {t:.2f}", fill=(255, 255, 255))
        frames.append(im)
    save_gif(boomerang(frames), "morph.gif", ms=60)


# ---------------------------------------------------------------- 3. Euler angles vs slerp
def gif_interpolation():
    k1 = np.array([60.0, 160.0, 110.0])
    q0, q1 = viewer.euler_to_quat(0, 0, 0), viewer.euler_to_quat(*k1)
    cam = viewer.Camera(eye=(0, 3.2, 9.5), target=(0, 0, 0), fovy=40)
    frames = []
    for t in np.linspace(0, 1, 40):
        scenes = []
        for R in (ex2.euler(*(t * k1)), ex2.quat_to_matrix(ex2.slerp(q0, q1, t))):
            s = viewer.Scene()
            s.add(viewer.airplane(), R)
            s.world_axes = 2.4
            s.object_axes.append((R, 2.0))
            scenes.append(s)
        fig = viewer.render_strip(scenes, cam, None, W=420, H=360,
                                  labels=["Euler angles (linear in the angles)", "slerp (shortest arc)"])
        for a_ in fig.axes:
            a_.text(0.5, 0.04, f"t = {t:.2f}", transform=a_.transAxes, ha="center", fontsize=9)
        frames.append(grab(fig))
    save_gif(boomerang(frames), "interpolation.gif", ms=50)


# ---------------------------------------------------------------- 4. world axes vs object axes
def gif_axes_modes():
    presses = "xxxxyyyyyy"
    vec = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    cam = viewer.Camera(eye=(0, 3.0, 9.0), target=(0, 0, 0), fovy=40)
    qw = qo = np.array([1.0, 0, 0, 0])
    frames = []
    steps = 7
    for key in presses:
        for k in range(1, steps + 1):
            dq = ex2.quat(vec[key], 15 * k / steps)
            scenes = []
            for mode, q0 in (("world", qw), ("object", qo)):
                q = ex2.quat_mul(dq, q0) if mode == "world" else ex2.quat_mul(q0, dq)
                R = ex2.quat_to_matrix(q)
                s = viewer.Scene()
                s.add(viewer.airplane(), R)
                s.world_axes = 2.6
                s.object_axes.append((R, 2.2))
                s.bold = mode
                scenes.append(s)
            fig = viewer.render_strip(scenes, cam, None, W=420, H=360,
                                      labels=["world axes: q ← qΔ q", "object axes: q ← q qΔ"])
            for a_ in fig.axes:
                a_.text(0.5, 0.04, f"pressing '{key}'  ({presses})", transform=a_.transAxes, ha="center", fontsize=9)
            frames.append(grab(fig))
        qw, qo = ex2.quat_mul(dq, qw), ex2.quat_mul(qo, dq)
    save_gif(frames, "axes-modes.gif", ms=50, hold=18)


# ---------------------------------------------------------------- 5. chase camera
def gif_chase():
    n = 64
    frames = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([4.2 * np.cos(a), 0.6 * np.sin(3 * a), 4.2 * np.sin(a)])
        vel = np.array([-4.2 * np.sin(a), 1.8 * np.cos(3 * a), 4.2 * np.cos(a)])
        acc = np.array([-4.2 * np.cos(a), -5.4 * np.sin(3 * a), -4.2 * np.sin(a)])
        f = vel / np.linalg.norm(vel)
        x = np.cross([0, 1.0, 0], f)
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, f, pos
        bank = -35 * (np.cross(vel, acc)[1] / np.linalg.norm(vel) ** 3) * 4.2
        M = M @ ex2.rotate((0, 0, 1), float(np.clip(bank, -35, 35)))
        V = ex2.chase_view(M)
        s = viewer.Scene()
        s.grid = -1.5
        s.add(viewer.airplane(), M)
        s.world_axes = 3.0
        for px, pz in ((-4, -4), (4, -4), (-4, 4), (4, 4), (0, 0)):    # pillars: something to fly around
            s.add(viewer.cube(0.5), ex2.translate(px, -1.25, pz))
        cam = viewer.Camera(eye=(0, 0, 0), target=(0, 0, -1), fovy=55, near=0.1, far=200.0, view=V)
        fig = viewer.render(s, cam, None, W=640, H=420)
        frames.append(grab(fig))
    save_gif(frames, "chase-camera.gif", ms=55)


if __name__ == "__main__":
    which = sys.argv[1:] or ["warp", "morph", "interpolation", "axes", "chase"]
    jobs = {"warp": gif_warp, "morph": gif_morph, "interpolation": gif_interpolation, "axes": gif_axes_modes,
            "chase": gif_chase}
    for w in which:
        jobs[w]()
