"""Exercise 2: the functions that the warp tool (Part A) and the viewer (Part B) call. Copy them into
your ex2.py and fill them in.

Part A conventions: images are float arrays with values in [0, 1], H x W (gray) or H x W x 3
(color). Pixel (row r, column c) has its center at (x, y) = (c, r). A list of N feature lines is an
N x 2 x 2 array: line i goes from P = lines[i, 0] to Q = lines[i, 1], each (x, y). Warping is
backward, and positions outside the source image are black.

Part B conventions: 4x4 matrices act on column vectors [x, y, z, 1]^T, so in A @ B the matrix B
applies first. Angles are in degrees. Eye coordinates are right-handed, and the camera looks down
its own -z axis. NDC is the cube [-1, 1]^3, with the near plane at z = -1. Raster coordinates:
x = W (x_ndc + 1) / 2, y = H (1 - y_ndc) / 2, depth (z_ndc + 1) / 2.
Quaternions are NumPy arrays [w, x, y, z], with the real part first.
"""
import numpy as np


# Part A
def _perp(v):
    """perp(x, y) = (-y, x), for arrays whose last axis is 2."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _sample(img, pts, interp="bilinear"):
    """Sample img at the N x 2 positions pts (x, y). Positions outside the image give black.
    Returns an array of shape (N,) for a gray image or (N, 3) for a color one."""
    img = np.asarray(img, dtype=float)
    pts = np.asarray(pts, dtype=float)
    H, W = img.shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    eps = 1e-9
    if interp == "nearest":
        xi = np.floor(x + 0.5).astype(int)
        yi = np.floor(y + 0.5).astype(int)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = img[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
    elif interp == "bilinear":
        inside = (x >= -eps) & (x <= W - 1 + eps) & (y >= -eps) & (y <= H - 1 + eps)
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        x0 = np.minimum(np.floor(x).astype(int), max(W - 2, 0))
        y0 = np.minimum(np.floor(y).astype(int), max(H - 2, 0))
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        fx, fy = x - x0, y - y0
        if img.ndim == 3:
            fx, fy = fx[:, None], fy[:, None]
        out = ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1])
               + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1]))
    else:
        raise ValueError(f"interp must be 'bilinear' or 'nearest', not {interp!r}")
    out = np.where(inside[:, None] if img.ndim == 3 else inside, out, 0.0)
    return out


def _pixel_grid(H, W):
    """The H*W x 2 array of pixel centers (x, y), in row-major order."""
    ys, xs = np.mgrid[0:H, 0:W]
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)


def source_positions(X, src_lines, dst_lines, a, b, p):
    # Beier-Neely: Nx2 output positions X (x, y) -> Nx2 source positions X'
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    src = np.asarray(src_lines, dtype=float).reshape(-1, 2, 2)
    dst = np.asarray(dst_lines, dtype=float).reshape(-1, 2, 2)
    P, Q = dst[:, 0], dst[:, 1]            # M x 2, the destination lines
    Pp, Qp = src[:, 0], src[:, 1]          # M x 2, the source lines
    tiny = 1e-12

    d = Q - P
    len2 = np.maximum((d ** 2).sum(axis=1), tiny)          # M
    length = np.sqrt(len2)
    dp = Qp - Pp
    lenp = np.sqrt(np.maximum((dp ** 2).sum(axis=1), tiny))

    rel = X[:, None, :] - P[None]                          # N x M x 2
    u = (rel * d[None]).sum(axis=2) / len2[None]           # N x M
    v = (rel * _perp(d)[None]).sum(axis=2) / length[None]  # N x M, signed distance

    # X'_i, the source position that line i alone gives
    Xi = Pp[None] + u[..., None] * dp[None] + v[..., None] * (_perp(dp) / lenp[:, None])[None]

    # distance from X to the destination segment
    to_P = np.sqrt((rel ** 2).sum(axis=2))
    to_Q = np.sqrt(((X[:, None, :] - Q[None]) ** 2).sum(axis=2))
    dist = np.where(u < 0, to_P, np.where(u > 1, to_Q, np.abs(v)))

    w = (length[None] ** p / (a + dist)) ** b              # N x M
    D = Xi - X[:, None, :]
    return X + (w[..., None] * D).sum(axis=1) / w.sum(axis=1)[:, None]


def warp_lines(img, src_lines, dst_lines, a, b, p, interp="bilinear"):
    # backward warp, output the size of img; interp: "bilinear" or "nearest"
    img = np.asarray(img, dtype=float)
    H, W = img.shape[:2]
    X = _pixel_grid(H, W)
    Xs = source_positions(X, src_lines, dst_lines, a, b, p)
    return _sample(img, Xs, interp).reshape(img.shape)


def morph(img0, img1, lines0, lines1, t, a, b, p):
    # the morph frame at time t in [0, 1]; img0 and img1 of the same size
    L0 = np.asarray(lines0, dtype=float)
    L1 = np.asarray(lines1, dtype=float)
    Lt = (1 - t) * L0 + t * L1
    warped0 = warp_lines(img0, L0, Lt, a, b, p)
    warped1 = warp_lines(img1, L1, Lt, a, b, p)
    return (1 - t) * warped0 + t * warped1


def fit_affine(src, dst):     # 3x3 A, dst ≈ A src (least squares)
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    S = np.hstack([src, np.ones((len(src), 1))])           # N x 3
    A2, *_ = np.linalg.lstsq(S, dst, rcond=None)           # 3 x 2, S @ A2 ≈ dst
    A = np.eye(3)
    A[:2, :] = A2.T
    return A


def fit_homography(src, dst):  # 3x3 H, dst ≈ H src, H[2,2] = 1
    # src, dst: Nx2 arrays of corresponding points (x, y)
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    x, y = src[:, 0], src[:, 1]
    xp, yp = dst[:, 0], dst[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    # x' (h6 x + h7 y + 1) = h0 x + h1 y + h2   (and the same for y')
    rows_x = np.stack([x, y, o, z, z, z, -xp * x, -xp * y], axis=1)
    rows_y = np.stack([z, z, z, x, y, o, -yp * x, -yp * y], axis=1)
    A = np.vstack([rows_x, rows_y])
    rhs = np.concatenate([xp, yp])
    h, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def warp_matrix(img, M, interp="bilinear"):
    # backward warp of img by the 3x3 M: output pixel X samples img at M^-1 X
    img = np.asarray(img, dtype=float)
    H, W = img.shape[:2]
    X = np.hstack([_pixel_grid(H, W), np.ones((H * W, 1))])   # (H*W) x 3 homogeneous
    Y = X @ np.linalg.inv(np.asarray(M, dtype=float)).T
    Xs = Y[:, :2] / Y[:, 2:3]
    return _sample(img, Xs, interp).reshape(img.shape)


# Part B
def translate(tx, ty, tz): ...      # 4x4 translation
def scale(sx, sy, sz): ...          # 4x4 scaling
def rotate(axis, angle): ...        # 4x4 rotation about a unit axis (Rodrigues)
def euler(ax, ay, az): ...          # 4x4: Rz(az) @ Ry(ay) @ Rx(ax)

def quat(axis, angle): ...          # unit quaternion of the rotation about a unit axis
def quat_mul(q1, q2): ...           # the product q1 q2
def quat_to_matrix(q): ...          # 4x4 rotation matrix of the unit quaternion q
def slerp(q0, q1, t): ...           # spherical linear interpolation, t in [0, 1]
    # along the great arc from q0 to q1 as given (do not replace q1 by -q1)

def look_at(eye, target, up): ...   # 4x4 view matrix, world -> eye
def ortho(l, r, b, t, near, far): ...           # 4x4, OpenGL convention
def perspective(fovy, aspect, near, far): ...   # 4x4, OpenGL convention
def viewport(W, H): ...             # 4x4: NDC -> raster x, y and depth
def project(M, pts): ...            # Nx3 -> Nx3: apply the 4x4 M, divide by w

def pixel_ray(M, x, y): ...         # (o, d): the ray of the world points that the 4x4 M
    # (viewport @ P @ V) takes to raster (x, y): o at depth 0, d unit, toward depth 1
def chase_view(M): ...              # 4x4 view matrix of the chase camera of B5
