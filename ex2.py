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
    flat = img.reshape(H * W, -1)                      # (H*W) x C, C = 1 for gray
    x, y = pts[:, 0], pts[:, 1]
    eps = 1e-9
    if interp == "nearest":
        xi = np.floor(x + 0.5).astype(np.intp)
        yi = np.floor(y + 0.5).astype(np.intp)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = flat[np.clip(yi, 0, H - 1) * W + np.clip(xi, 0, W - 1)]
        out *= inside[:, None]
    elif interp == "bilinear":
        inside = (x >= -eps) & (x <= W - 1 + eps) & (y >= -eps) & (y <= H - 1 + eps)
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        x0 = np.minimum(x.astype(np.intp), max(W - 2, 0))   # x >= 0, so astype == floor
        y0 = np.minimum(y.astype(np.intp), max(H - 2, 0))
        fx = x - x0
        fy = y - y0
        i00 = y0 * W + x0
        di = 1 if W > 1 else 0
        dj = W if H > 1 else 0
        # separable weights, with the "inside" mask folded in so no extra pass is needed
        w11 = fx * fy
        w10 = fy - w11                  # (1-fx) * fy
        w01 = fx - w11                  # fx * (1-fy)
        w00 = 1 - fx - fy + w11         # (1-fx) * (1-fy)
        m = inside.astype(float)
        out = (flat[i00] * (w00 * m)[:, None] + flat[i00 + di] * (w01 * m)[:, None]
               + flat[i00 + dj] * (w10 * m)[:, None] + flat[i00 + dj + di] * (w11 * m)[:, None])
    else:
        raise ValueError(f"interp must be 'bilinear' or 'nearest', not {interp!r}")
    return out.reshape(len(pts)) if img.ndim == 2 else out


_GRID_CACHE = {}


def _pixel_grid(H, W):
    """The H*W x 2 array of pixel centers (x, y), in row-major order (cached; do not modify)."""
    g = _GRID_CACHE.get((H, W))
    if g is None:
        ys, xs = np.mgrid[0:H, 0:W]
        g = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
        g.flags.writeable = False
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[(H, W)] = g
    return g


def source_positions(X, src_lines, dst_lines, a, b, p):
    # Beier-Neely: Nx2 output positions X (x, y) -> Nx2 source positions X'
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    src = np.asarray(src_lines, dtype=float).reshape(-1, 2, 2)
    dst = np.asarray(dst_lines, dtype=float).reshape(-1, 2, 2)
    tiny = 1e-12
    x, y = X[:, 0:1], X[:, 1:2]                              # N x 1

    # per-line quantities, each as a 1 x M row
    Px, Py = dst[:, 0, 0][None], dst[:, 0, 1][None]
    Qx, Qy = dst[:, 1, 0][None], dst[:, 1, 1][None]
    dx, dy = Qx - Px, Qy - Py
    len2 = np.maximum(dx * dx + dy * dy, tiny)
    length = np.sqrt(len2)
    Ppx, Ppy = src[:, 0, 0][None], src[:, 0, 1][None]
    dpx, dpy = src[:, 1, 0][None] - Ppx, src[:, 1, 1][None] - Ppy
    lenp = np.sqrt(np.maximum(dpx * dpx + dpy * dpy, tiny))

    rx, ry = x - Px, y - Py                                  # N x M
    u = (rx * dx + ry * dy) / len2
    v = (-rx * dy + ry * dx) / length                        # signed distance, perp(d) = (-dy, dx)

    # X'_i - X: the displacement that line i alone gives (perp(dp) = (-dpy, dpx))
    Dx = Ppx + u * dpx - v * dpy / lenp - x
    Dy = Ppy + u * dpy + v * dpx / lenp - y

    # distance from X to the destination segment: |v| if 0 <= u <= 1, else to the nearer endpoint.
    # Both cases are the distance to the closest point P + clip(u, 0, 1) d of the segment.
    cu = np.clip(u, 0.0, 1.0)
    dist = np.hypot(rx - cu * dx, ry - cu * dy)

    base = length ** p / (a + dist)                          # N x M
    w = base * base if b == 2 else base ** b
    sw = w.sum(axis=1)
    return np.stack([X[:, 0] + (w * Dx).sum(axis=1) / sw,
                     X[:, 1] + (w * Dy).sum(axis=1) / sw], axis=1)


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
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def translate(tx, ty, tz):       # 4x4 translation
    T = np.eye(4)
    T[:3, 3] = (tx, ty, tz)
    return T


def scale(sx, sy, sz):           # 4x4 scaling
    return np.diag([sx, sy, sz, 1.0])


def rotate(axis, angle):         # 4x4 rotation about a unit axis (Rodrigues)
    k = _unit(axis)
    th = np.radians(angle)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])                      # K v = k x v
    R = np.eye(4)
    R[:3, :3] = c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)
    return R


def euler(ax, ay, az):           # 4x4: Rz(az) @ Ry(ay) @ Rx(ax)
    return rotate((0, 0, 1), az) @ rotate((0, 1, 0), ay) @ rotate((1, 0, 0), ax)


def quat(axis, angle):           # unit quaternion of the rotation about a unit axis
    half = np.radians(angle) / 2
    return np.concatenate([[np.cos(half)], np.sin(half) * _unit(axis)])


def quat_mul(q1, q2):            # the product q1 q2
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_to_matrix(q):           # 4x4 rotation matrix of the unit quaternion q
    w, x, y, z = _unit(q)
    R = np.eye(4)
    R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return R


def slerp(q0, q1, t):            # spherical linear interpolation, t in [0, 1]
    # along the great arc from q0 to q1 as given (do not replace q1 by -q1)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    omega = np.arccos(dot)                      # the angle of the arc, in [0, pi]
    sin_o = np.sin(omega)
    if sin_o < 1e-8:
        if dot > 0:                             # the same quaternion: nothing to interpolate
            q = (1 - t) * q0 + t * q1
            return q / np.linalg.norm(q)
        # antipodal: every great circle through q0 and -q1 is an arc; take one of them
        e = np.zeros(4)
        e[np.argmin(np.abs(q0))] = 1.0
        r = _unit(e - (e @ q0) * q0)
        return np.cos(np.pi * t) * q0 + np.sin(np.pi * t) * r
    return (np.sin((1 - t) * omega) * q0 + np.sin(t * omega) * q1) / sin_o


def look_at(eye, target, up):    # 4x4 view matrix, world -> eye
    eye = np.asarray(eye, dtype=float)
    f = _unit(np.asarray(target, dtype=float) - eye)           # viewing direction (the camera's -z)
    s = np.cross(f, np.asarray(up, dtype=float))               # the camera's right
    if np.linalg.norm(s) < 1e-12:                              # up parallel to the viewing direction
        s = np.cross(f, [1.0, 0.0, 0.0] if abs(f[0]) < 0.9 else [0.0, 0.0, 1.0])
    s = _unit(s)
    u = np.cross(s, f)                                         # the camera's up
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = s, u, -f
    V[:3, 3] = -V[:3, :3] @ eye
    return V


def ortho(l, r, b, t, near, far):            # 4x4, OpenGL convention
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)],
                     [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                     [0, 0, 0, 1.0]])


def perspective(fovy, aspect, near, far):    # 4x4, OpenGL convention
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])


def viewport(W, H):              # 4x4: NDC -> raster x, y and depth
    return np.array([[W / 2, 0, 0, W / 2],
                     [0, -H / 2, 0, H / 2],
                     [0, 0, 0.5, 0.5],
                     [0, 0, 0, 1.0]])


def project(M, pts):             # Nx3 -> Nx3: apply the 4x4 M, divide by w
    pts = np.asarray(pts, dtype=float)
    h = np.hstack([pts.reshape(-1, 3), np.ones((pts.size // 3, 1))]) @ np.asarray(M, dtype=float).T
    return (h[:, :3] / h[:, 3:4]).reshape(pts.shape)


def pixel_ray(M, x, y):          # (o, d): the ray of the world points that the 4x4 M
    # (viewport @ P @ V) takes to raster (x, y): o at depth 0, d unit, toward depth 1
    Minv = np.linalg.inv(np.asarray(M, dtype=float))
    ends = []
    for depth in (0.0, 1.0):
        h = Minv @ np.array([x, y, depth, 1.0])
        ends.append(h[:3] / h[3])
    o = ends[0]
    return o, _unit(ends[1] - o)


def chase_view(M):               # 4x4 view matrix of the chase camera of B5
    M = np.asarray(M, dtype=float)
    eye = (M @ np.array([0.0, 2.0, -8.0, 1.0]))[:3]      # a point: behind and above the airplane
    target = (M @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]    # a point: the airplane's center
    up = M[:3, :3] @ np.array([0.0, 1.0, 0.0])           # a direction: the airplane's +y
    return look_at(eye, target, up)
