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
def source_positions(X, src_lines, dst_lines, a, b, p): ...
    # Beier-Neely: Nx2 output positions X (x, y) -> Nx2 source positions X'
def warp_lines(img, src_lines, dst_lines, a, b, p, interp="bilinear"): ...
    # backward warp, output the size of img; interp: "bilinear" or "nearest"
def morph(img0, img1, lines0, lines1, t, a, b, p): ...
    # the morph frame at time t in [0, 1]; img0 and img1 of the same size
def fit_affine(src, dst): ...     # 3x3 A, dst ≈ A src (least squares)
def fit_homography(src, dst): ... # 3x3 H, dst ≈ H src, H[2,2] = 1
    # src, dst: Nx2 arrays of corresponding points (x, y)
def warp_matrix(img, M, interp="bilinear"): ...
    # backward warp of img by the 3x3 M: output pixel X samples img at M^-1 X


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
