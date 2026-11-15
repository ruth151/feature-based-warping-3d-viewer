"""My own sanity tests for Part A of ex2.py. Run: python test_partA.py"""
import time
import numpy as np
import ex2

SRC = np.array([[[64, 128], [192, 128]]], float)
rng = np.random.default_rng(0)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


def line(p, q):
    return np.array([[p, q]], float)


# ---------------------------------------------------------------- source_positions
X = rng.uniform(0, 256, (50, 2))
check("sp: identical lines -> X' = X",
      close(ex2.source_positions(X, SRC, SRC.copy(), 1, 2, 0.5), X))

dst = line((84, 148), (212, 148))
check("sp: translation by (20,20) -> X' = X - (20,20)",
      close(ex2.source_positions(X, SRC, dst, 1, 2, 0.5), X - 20))

# rotation by 90 deg about the midpoint (128,128): P'->Q' is +x, P->Q is +y.
# A point at (128+s, 128+t) in dst coords maps to source (128+t, 128-s)... derive: u,v relative to dst
dst = line((128, 64), (128, 192))
out = ex2.source_positions(np.array([[128., 64.], [128., 192.], [128., 128.]]), SRC, dst, 1, 2, 0.5)
check("sp: rotated line maps P->P', Q->Q', mid->mid",
      close(out, [[64, 128], [192, 128], [128, 128]]))
# v>0 side: perp(Q-P)=perp(0,128)=(-128,0)/128 -> (-1,0): point (127,128) has v=+1
out = ex2.source_positions(np.array([[127., 128.]]), SRC, dst, 1, 2, 0.5)
check("sp: rotated line, off-line point has u=.5, v=+1 -> (128,129)",
      close(out, [[128, 129]]))

dst = line((0, 128), (256, 128))
out = ex2.source_positions(np.array([[128., 64.], [160., 64.]]), SRC, dst, 1, 2, 0.5)
# doubled line: X' = c + (X-c)/2 for u along, v unchanged: (128,64) -> u=.5,v=-64 -> (128,128-64)? perp((256,0))=(0,256)/256=(0,1); v=(0,-64).(0,1)=-64
check("sp: doubled line, hand-computed (128,64)->(128,64), (160,64)->(144,64)",
      close(out, [[128, 64], [144, 64]]))

# crossing lines A3: C is fixed, and each is half as long
dst2 = np.array([[[96, 128], [160, 128]], [[128, 96], [128, 160]]], float)
src2 = np.array([[[112, 128], [144, 128]], [[128, 112], [128, 144]]], float)
out = ex2.source_positions(np.array([[128., 128.]]), src2, dst2, 1, 2, 0.5)
check("sp: crossing lines, C fixed", close(out, [[128, 128]]))
out = ex2.source_positions(np.array([[140., 128.]]), src2, dst2, 1, 2, 0.5)
check("sp: crossing lines, (140,128) maps to x'<140 (shrunk toward C)", out[0, 0] < 140 and abs(out[0, 1] - 128) < 1e-6)

# two lines, moved first line: endpoints of the lines map exactly
src3 = np.array([[[64, 64], [192, 64]], [[64, 192], [192, 192]]], float)
dst3 = np.array([[[64, 96], [192, 96]], [[64, 192], [192, 192]]], float)
out = ex2.source_positions(np.array([[128., 96.]]), src3, dst3, 1, 2, 0.5)
check("sp: two lines, point on moved line comes (almost) from the original line",
      abs(out[0, 1] - 64) < 2.0)
out = ex2.source_positions(np.array([[128., 120.], [128., 144.]]), src3, dst3, 1, 2, 0.5)
check("sp: two lines, displacement y'-y is between -32 and 0 and monotone",
      -32 <= out[0, 1] - 120 <= 0 and -32 <= out[1, 1] - 144 <= 0 and abs(out[0, 1] - 120) > abs(out[1, 1] - 144))

check("sp: shape (N,2), single point (1,2)",
      ex2.source_positions(X, SRC, SRC, 1, 2, .5).shape == (50, 2) and
      ex2.source_positions(X[:1], SRC, SRC, 1, 2, .5).shape == (1, 2))
Xc = X.copy(); S = SRC.copy(); D = dst.copy()
ex2.source_positions(X, S, D, 1, 2, .5)
check("sp: inputs unchanged", np.array_equal(X, Xc) and np.array_equal(S, SRC) and np.array_equal(D, dst))

# ---------------------------------------------------------------- fit_affine / fit_homography
A_true = np.array([[1.2, -0.3, 10.], [0.4, 0.9, -5.], [0, 0, 1]])
pts = rng.uniform(0, 200, (10, 2))
ph = np.c_[pts, np.ones(10)]
dstp = (ph @ A_true.T)[:, :2]
check("fit_affine: recovers exact affine", close(ex2.fit_affine(pts, dstp), A_true, 1e-6))
noisy = dstp + rng.normal(0, 1, dstp.shape)
A = ex2.fit_affine(pts, noisy)
ref = np.linalg.lstsq(ph, noisy, rcond=None)[0].T
check("fit_affine: least squares matches lstsq on noisy data", close(A[:2], ref, 1e-6) and close(A[2], [0, 0, 1]))
check("fit_affine: 3 points exact", close(ex2.fit_affine(pts[:3], dstp[:3]), A_true, 1e-6))

H_true = np.array([[1.1, 0.1, 5.], [-0.05, 0.95, 3.], [1e-3, -5e-4, 1.]])
q = ph @ H_true.T
dsth = q[:, :2] / q[:, 2:]
H = ex2.fit_homography(pts, dsth)
check("fit_homography: recovers exact homography (10 pts)", close(H, H_true, 1e-5))
check("fit_homography: H[2,2] == 1", abs(H[2, 2] - 1) < 1e-12)
H4 = ex2.fit_homography(pts[:4], dsth[:4])
check("fit_homography: 4 pts exact", close(H4, H_true, 1e-5))
trap_src = np.array([[64, 96], [192, 96], [64, 192], [192, 192]], float)
trap_dst = np.array([[96, 96], [160, 96], [64, 192], [192, 192]], float)
H = ex2.fit_homography(trap_src, trap_dst)
q = np.c_[trap_src, np.ones(4)] @ H.T
check("fit_homography: trapezoid endpoints reproduced", close(q[:, :2] / q[:, 2:], trap_dst, 1e-6))

# ---------------------------------------------------------------- warp_matrix
img = rng.uniform(0, 1, (20, 30))
imgc = rng.uniform(0, 1, (20, 30, 3))
I3 = np.eye(3)
check("warp_matrix: identity (bilinear, gray)", close(ex2.warp_matrix(img, I3), img))
check("warp_matrix: identity (nearest, color)", close(ex2.warp_matrix(imgc, I3, "nearest"), imgc))
T = np.array([[1, 0, 3.], [0, 1, 2.], [0, 0, 1]])
for interp in ("nearest", "bilinear"):
    out = ex2.warp_matrix(imgc, T, interp)
    check(f"warp_matrix: integer translation, {interp}",
          out.shape == imgc.shape and close(out[2:, 3:], imgc[:-2, :-3]) and
          np.all(out[:2] == 0) and np.all(out[:, :3] == 0))
T2 = np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]])
out = ex2.warp_matrix(img, T2, "bilinear")
check("warp_matrix: half-pixel shift bilinear = average of neighbours",
      close(out[:, 1:], (img[:, :-1] + img[:, 1:]) / 2))
S = np.diag([2., 2., 1.])
out = ex2.warp_matrix(img, S, "nearest")
check("warp_matrix: scale 2 nearest, out(2c,2r)=img(c,r)",
      close(out[::2, ::2][:10, :15], img[:10, :15]) and close(out[4, 6], img[2, 3]))
# a homography with a true perspective part
Hm = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, 1.]])
out = ex2.warp_matrix(img, Hm, "bilinear")
Minv = np.linalg.inv(Hm); v = Minv @ [10, 5, 1]; v = v[:2] / v[2]
check("warp_matrix: homography output pixel (10,5) samples M^-1 X",
      abs(out[5, 10] - ex2._sample(img, v[None], "bilinear")[0]) < 1e-9)

# ---------------------------------------------------------------- warp_lines / morph
big = rng.uniform(0, 1, (64, 64, 3))
L = line((10, 20), (50, 40))
check("warp_lines: identical lines -> identity", close(ex2.warp_lines(big, L, L.copy(), 1, 2, .5), big))
check("warp_lines: identical lines -> identity (nearest)", close(ex2.warp_lines(big, L, L.copy(), 1, 2, .5, "nearest"), big))
Ld = L + [3, 2]
out = ex2.warp_lines(big, L, Ld, 1, 2, .5, "nearest")
check("warp_lines: single translated line = integer shift",
      close(out[2:, 3:], big[:-2, :-3]))
g = ex2.warp_lines(rng.uniform(0, 1, (30, 40)), L, Ld, 1, 2, .5)
check("warp_lines: gray in -> gray out, same size", g.shape == (30, 40))
check("warp_lines: output within [0,1]", out.min() >= 0 and out.max() <= 1 and g.min() >= 0 and g.max() <= 1)

img1 = rng.uniform(0, 1, (64, 64, 3))
L0 = np.array([[[10, 10], [50, 12]], [[20, 40], [40, 50]]], float)
L1 = np.array([[[14, 16], [48, 20]], [[22, 44], [44, 46]]], float)
check("morph: t=0 gives img0", close(ex2.morph(big, img1, L0, L1, 0.0, 1, 2, .5), big))
check("morph: t=1 gives img1", close(ex2.morph(big, img1, L0, L1, 1.0, 1, 2, .5), img1))
m = ex2.morph(big, img1, L0, L1, .5, 1, 2, .5)
check("morph: t=.5 shape/range", m.shape == big.shape and 0 <= m.min() and m.max() <= 1)
check("morph: identical lines t=.5 = cross-dissolve", close(ex2.morph(big, img1, L0, L0, .5, 1, 2, .5), (big + img1) / 2))

# ---------------------------------------------------------------- speed (no pixel loops)
im = rng.uniform(0, 1, (256, 256, 3))
La = np.array([[[64, 64], [192, 64]], [[64, 192], [192, 192]], [[10, 10], [100, 200]]], float)
Lb = La + rng.uniform(-10, 10, La.shape)
t0 = time.time(); ex2.warp_lines(im, La, Lb, 1, 2, .5); dt = time.time() - t0
check(f"speed: warp_lines 256x256, 3 lines took {dt:.2f}s (< 1s)", dt < 1.0)
t0 = time.time(); ex2.warp_matrix(im, A_true, "bilinear"); dt = time.time() - t0
check(f"speed: warp_matrix 256x256 took {dt:.2f}s (< 0.5s)", dt < 0.5)

print("\nFAILED:" if fails else "\nALL PASSED", *fails, sep="\n  ")
raise SystemExit(1 if fails else 0)
