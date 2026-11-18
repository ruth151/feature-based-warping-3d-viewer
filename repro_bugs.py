"""B6: reproduce bug1.png, bug2.png and bug3.png by planting one error in one function of ex2.

Run: python repro_bugs.py   (writes bugs_repro/repro1.png ... repro3.png and compares them with bugs/bug*.png)
The errors are planted by replacing the correct function of ex2 with a copy that has the error, which
is the same as editing a copy of ex2.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import ex2
import viewer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bugs_repro")


def start_view(file):
    """The viewer's start view: azimuth 35, elevation 20, distance 10, fovy 40, axes and ground grid."""
    az, el, d = np.radians(35), np.radians(20), 10.0
    eye = d * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    s = viewer.Scene()
    s.grid = -1.5
    M = np.eye(4)
    s.add(viewer.airplane(), M)
    s.world_axes = 3.0
    s.object_axes.append((M, 2.6))
    viewer.render(s, viewer.Camera(eye, (0, 0, 0), (0, 1, 0), 40.0, 0.1, 200.0, False), file)
    plt.close("all")


# ---- the three planted errors ----------------------------------------------------------------
_perspective, _look_at, _viewport = ex2.perspective, ex2.look_at, ex2.viewport


def perspective_bug(fovy, aspect, near, far):          # bug 1: w = +z instead of -z (P[3, 2] = +1)
    P = _perspective(fovy, aspect, near, far)
    P[3, 2] = +1.0
    return P


def look_at_bug(eye, target, up):                      # bug 2: the cross product in the wrong order
    eye = np.asarray(eye, dtype=float)
    f = ex2._unit(np.asarray(target, dtype=float) - eye)
    s = ex2._unit(np.cross(up, f))                     # correct: np.cross(f, up)
    u = np.cross(s, f)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = s, u, -f
    V[:3, 3] = -V[:3, :3] @ eye
    return V


def viewport_bug(W, H):                                # bug 3: y is not flipped (+H/2 instead of -H/2)
    return np.array([[W / 2, 0, 0, W / 2], [0, +H / 2, 0, H / 2], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]])


BUGS = {1: ("perspective", perspective_bug, _perspective),
        2: ("look_at", look_at_bug, _look_at),
        3: ("viewport", viewport_bug, _viewport)}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, (name, bad, good) in BUGS.items():
        setattr(ex2, name, bad)
        out = os.path.join(OUT, f"repro{i}.png")
        start_view(out)
        setattr(ex2, name, good)
        a = np.asarray(Image.open(out).convert("RGB"), float)
        b = np.asarray(Image.open(os.path.join(HERE, "bugs", f"bug{i}.png")).convert("RGB"), float)
        print(f"bug{i}: error planted in {name}(); mean grey-level difference to bugs/bug{i}.png = {np.abs(a - b).mean():.3f}")
