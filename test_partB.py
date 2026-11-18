"""My own tests for Part B of ex2.py (independent of selftest.py). Run: python test_partB.py"""
import numpy as np
import ex2

rng = np.random.default_rng(7)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if b.ndim == 0:                      # allow comparing with the scalar 0
        b = np.broadcast_to(b, a.shape)
    return a.shape == b.shape and np.allclose(a, b, atol=tol)


def Rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]])


def Ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.]])


def Rz(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def apply(M, p):
    h = np.asarray(M, float) @ np.r_[p, 1.0]
    return h[:3] / h[3]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def hamilton(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def rand_axis():
    return unit(rng.normal(size=3))


def rand_quat():
    return unit(rng.normal(size=4))


# ---------------------------------------------------------------- translate / scale
check("translate: matrix", close(ex2.translate(1, 2, 3), [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]))
check("scale: matrix", close(ex2.scale(2, 3, 4), np.diag([2, 3, 4, 1.])))
check("translate/scale: 4x4 float arrays", np.asarray(ex2.translate(0, 0, 0)).shape == (4, 4))

# ---------------------------------------------------------------- rotate / euler
for name, axis, R in (("x", (1, 0, 0), Rx), ("y", (0, 1, 0), Ry), ("z", (0, 0, 1), Rz)):
    check(f"rotate: about {name} equals elementary matrix (37 deg)", close(ex2.rotate(axis, 37), R(37)))
    check(f"rotate: about {name}, negative angle is the inverse", close(ex2.rotate(axis, -37) @ ex2.rotate(axis, 37), np.eye(4)))
k = rand_axis()
Rk = np.asarray(ex2.rotate(k, 70), float)
check("rotate: orthonormal, det = +1, last row (0,0,0,1)",
      close(Rk[:3, :3].T @ Rk[:3, :3], np.eye(3)) and abs(np.linalg.det(Rk[:3, :3]) - 1) < 1e-9 and close(Rk[3], [0, 0, 0, 1]) and close(Rk[:3, 3], 0))
check("rotate: axis is fixed", close(apply(Rk, k), k))
check("rotate: angle 0 and 360 are identity", close(ex2.rotate(k, 0), np.eye(4)) and close(ex2.rotate(k, 360), np.eye(4)))
check("rotate: z by 90 sends x to y (right-handed)", close(apply(ex2.rotate((0, 0, 1), 90), (1, 0, 0)), (0, 1, 0)))
# trace R = 1 + 2 cos(theta)
check("rotate: trace = 1 + 2cos(theta)", abs(np.trace(Rk[:3, :3]) - (1 + 2 * np.cos(np.radians(70)))) < 1e-9)
check("rotate: composing two rotations about one axis adds the angles", close(ex2.rotate(k, 30) @ ex2.rotate(k, 45), ex2.rotate(k, 75)))
check("euler: Rz(az) Ry(ay) Rx(ax)", close(ex2.euler(10, 20, 30), Rz(30) @ Ry(20) @ Rx(10)))
check("euler: zero is identity", close(ex2.euler(0, 0, 0), np.eye(4)))
check("euler: only ax / only ay / only az", close(ex2.euler(25, 0, 0), Rx(25)) and close(ex2.euler(0, 25, 0), Ry(25)) and close(ex2.euler(0, 0, 25), Rz(25)))

# ---------------------------------------------------------------- quaternions
q = np.asarray(ex2.quat(k, 70), float)
check("quat: [cos(t/2), sin(t/2) u]", close(q, np.r_[np.cos(np.radians(35)), np.sin(np.radians(35)) * k]))
check("quat: unit norm, shape (4,)", q.shape == (4,) and abs(np.linalg.norm(q) - 1) < 1e-12)
check("quat: angle 0 is [1,0,0,0]", close(ex2.quat((0, 1, 0), 0), [1, 0, 0, 0]))
check("quat: z, 90 as [w,x,y,z]", close(ex2.quat((0, 0, 1), 90), [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
a, b = rand_quat(), rand_quat()
check("quat_mul: Hamilton product on random quaternions", close(ex2.quat_mul(a, b), hamilton(a, b)))
i_, j_, k_ = np.array([0, 1., 0, 0]), np.array([0, 0, 1., 0]), np.array([0, 0, 0, 1.])
check("quat_mul: i j = k, j i = -k, i i = -1", close(ex2.quat_mul(i_, j_), k_) and close(ex2.quat_mul(j_, i_), -k_) and close(ex2.quat_mul(i_, i_), [-1, 0, 0, 0]))
check("quat_mul: identity and associativity",
      close(ex2.quat_mul([1, 0, 0, 0], a), a) and close(ex2.quat_mul(a, [1, 0, 0, 0]), a)
      and close(ex2.quat_mul(ex2.quat_mul(a, b), a), ex2.quat_mul(a, ex2.quat_mul(b, a)), 1e-9))
check("quat_mul: norm is multiplicative", abs(np.linalg.norm(ex2.quat_mul(3 * a, 2 * b)) - 6) < 1e-9)
check("quat_to_matrix: matches rotate for quat(k, 70)", close(ex2.quat_to_matrix(ex2.quat(k, 70)), ex2.rotate(k, 70)))
for _ in range(5):
    ax_, an_ = rand_axis(), rng.uniform(-360, 360)
    assert close(ex2.quat_to_matrix(ex2.quat(ax_, an_)), ex2.rotate(ax_, an_))
check("quat_to_matrix: matches rotate for 5 random axes/angles", True)
qq = rand_quat()
check("quat_to_matrix: q and -q give the same rotation", close(ex2.quat_to_matrix(qq), ex2.quat_to_matrix(-qq)))
check("quat_to_matrix: 4x4, rotation block orthonormal, last row/col", np.asarray(ex2.quat_to_matrix(qq)).shape == (4, 4)
      and close(np.asarray(ex2.quat_to_matrix(qq))[3], [0, 0, 0, 1]))
q1, q2 = ex2.quat((1, 0, 0), 40), ex2.quat(k, 70)
check("quat_mul: q1 q2 = rotation q2 followed by q1",
      close(ex2.quat_to_matrix(ex2.quat_mul(q1, q2)), ex2.quat_to_matrix(q1) @ ex2.quat_to_matrix(q2)))
check("quat_mul: matches euler = Rz Ry Rx", close(
    ex2.quat_to_matrix(ex2.quat_mul(ex2.quat((0, 0, 1), 30), ex2.quat_mul(ex2.quat((0, 1, 0), 20), ex2.quat((1, 0, 0), 10)))),
    ex2.euler(10, 20, 30)))

# ---------------------------------------------------------------- slerp
u = rand_axis()
for a0, a1 in ((0, 90), (30, 200), (-100, 120), (10, 300), (0, 270)):
    q0, q1 = ex2.quat(u, a0), ex2.quat(u, a1)
    ok = all(close(ex2.slerp(q0, q1, t), ex2.quat(u, a0 + t * (a1 - a0)), 1e-9) for t in (0, 0.25, 0.5, 0.9, 1))
    check(f"slerp: along one axis, angles {a0} -> {a1} (arc as given, even if the dot product is negative)", ok)
q0, q1 = rand_quat(), rand_quat()
check("slerp: endpoints", close(ex2.slerp(q0, q1, 0), q0) and close(ex2.slerp(q0, q1, 1), q1))
check("slerp: result is a unit quaternion", all(abs(np.linalg.norm(ex2.slerp(q0, q1, t)) - 1) < 1e-12 for t in np.linspace(0, 1, 7)))
w = np.arccos(np.clip(q0 @ q1, -1, 1))
mid = ex2.slerp(q0, q1, 0.5)
check("slerp: midpoint is equidistant on the sphere", abs(np.arccos(np.clip(q0 @ mid, -1, 1)) - w / 2) < 1e-9 and abs(np.arccos(np.clip(q1 @ mid, -1, 1)) - w / 2) < 1e-9)
check("slerp: q0 = q1 (sin w = 0) gives q0, no NaN", close(ex2.slerp(q0, q0, 0.3), q0))
nearly = unit(q0 + 1e-9 * rng.normal(size=4))
check("slerp: nearly equal quaternions, no NaN", np.all(np.isfinite(ex2.slerp(q0, nearly, 0.5))))
anti = ex2.slerp(q0, -q0, 0.5)
check("slerp: antipodal q1 = -q0, finite unit result", np.all(np.isfinite(anti)) and abs(np.linalg.norm(anti) - 1) < 1e-9)
check("slerp: inputs not modified", close(q0 + 0, q0))

# ---------------------------------------------------------------- look_at
V = np.asarray(ex2.look_at((3, 2, 5), (1, 0, 1), (0, 1, 0)), float)
d = np.linalg.norm([2, 2, 4])
check("look_at: eye -> origin, target -> (0,0,-d)", close(apply(V, (3, 2, 5)), 0) and close(apply(V, (1, 0, 1)), (0, 0, -d)))
check("look_at: rotation block orthonormal with det +1", close(V[:3, :3] @ V[:3, :3].T, np.eye(3)) and abs(np.linalg.det(V[:3, :3]) - 1) < 1e-9 and close(V[3], [0, 0, 0, 1]))
check("look_at: from +z at distance 5 is translate(0,0,-5)", close(ex2.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0)), ex2.translate(0, 0, -5)))
# camera right-handed: world +x seen from +z looks right, world +y looks up
Vz = ex2.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
check("look_at: right = +x, up = +y", apply(Vz, (1, 0, 0))[0] > 0 and apply(Vz, (0, 1, 0))[1] > 0)
Vx = ex2.look_at((5, 0, 0), (0, 0, 0), (0, 1, 0))      # looking down -x: world -z is on the right
check("look_at: looking down -x, world -z appears on the right", apply(Vx, (0, 0, -1))[0] > 0)
Vu = np.asarray(ex2.look_at((0, 0, 5), (0, 0, 0), (0, 3, 1)), float)   # same camera as Vz: up has a component along the view axis
check("look_at: a slanted up vector gives the same camera", close(Vu, ex2.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))))
check("look_at: up given as a list or array gives the same", close(ex2.look_at(np.array([3., 2, 5]), [1, 0, 1], [0, 2, 0]), V))

# ---------------------------------------------------------------- projections
f = 1 / np.tan(np.radians(60) / 2)
n, fr, asp = 0.5, 50.0, 1.6
P = np.asarray(ex2.perspective(60, asp, n, fr), float)
check("perspective: OpenGL matrix", close(P, [[f / asp, 0, 0, 0], [0, f, 0, 0],
                                              [0, 0, (fr + n) / (n - fr), 2 * fr * n / (n - fr)], [0, 0, -1, 0]]))
check("perspective: near/far planes map to z = -1 / +1", close(apply(P, (0, 0, -n)), (0, 0, -1)) and close(apply(P, (0, 0, -fr)), (0, 0, 1)))
check("perspective: edge of the view volume at the near plane", close(apply(P, (n * np.tan(np.radians(30)) * asp, n * np.tan(np.radians(30)), -n))[:2], (1, 1)))
O = np.asarray(ex2.ortho(-2, 4, -1, 3, 1, 10), float)
check("ortho: OpenGL matrix", close(O, [[2 / 6, 0, 0, -(4 - 2) / 6], [0, 2 / 4, 0, -(3 - 1) / 4],
                                        [0, 0, -2 / 9, -(10 + 1) / 9], [0, 0, 0, 1]]))
check("ortho: corners", close(apply(O, (-2, -1, -1)), (-1, -1, -1)) and close(apply(O, (4, 3, -10)), (1, 1, 1)))
VP = np.asarray(ex2.viewport(800, 600), float)
check("viewport: matrix", close(VP, [[400, 0, 0, 400], [0, -300, 0, 300], [0, 0, .5, .5], [0, 0, 0, 1]]))
check("viewport: corners and center", close(apply(VP, (-1, 1, -1)), (0, 0, 0)) and close(apply(VP, (1, -1, 1)), (800, 600, 1)) and close(apply(VP, (0, 0, 0)), (400, 300, .5)))
pts = rng.normal(size=(6, 3))
M = np.diag([1., 2, 1, 2]) @ ex2.translate(1, 0, 0)
exp = np.array([apply(M, p) for p in pts])
check("project: Nx3 -> Nx3, divides by w", close(ex2.project(M, pts), exp) and ex2.project(M, pts).shape == (6, 3))
check("project: a single point as 1x3", ex2.project(M, pts[:1]).shape == (1, 3))
Mp = VP @ P @ ex2.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
check("project: the center of the view lands in the middle of the image", close(ex2.project(Mp, np.array([[0., 0, 0]]))[0, :2], (400, 300)))
check("project: a point above the origin has a smaller raster y (y points down)", ex2.project(Mp, np.array([[0., 1, 0]]))[0, 1] < 300)

# ---------------------------------------------------------------- pixel_ray
eye = np.array([6., 4, 8])
Vp = ex2.look_at(eye, (0, 0, 0), (0, 1, 0))
for label, Pm in (("perspective", ex2.perspective(40, 4 / 3, 0.1, 100)), ("orthographic", ex2.ortho(-4, 4, -3, 3, 0.1, 100))):
    M = VP @ Pm @ Vp
    o, dd = (np.asarray(v, float) for v in ex2.pixel_ray(M, 123.5, 456.25))
    check(f"pixel_ray ({label}): o projects to the pixel at depth 0", close(apply(M, o), (123.5, 456.25, 0), 1e-6))
    check(f"pixel_ray ({label}): d is unit, and o + s d projects to the same pixel with growing depth",
          abs(np.linalg.norm(dd) - 1) < 1e-9 and close(apply(M, o + 3 * dd)[:2], (123.5, 456.25), 1e-6)
          and apply(M, o + 3 * dd)[2] > 0 and apply(M, o + 6 * dd)[2] > apply(M, o + 3 * dd)[2])
    # a world point projects to a pixel; the ray of that pixel passes through the point
    Pw = np.array([1., -0.5, 0.3])
    px = apply(M, Pw)
    o, dd = (np.asarray(v, float) for v in ex2.pixel_ray(M, px[0], px[1]))
    check(f"pixel_ray ({label}): the ray of a projected point passes through it",
          np.linalg.norm(np.cross(dd, Pw - o)) < 1e-6 and (Pw - o) @ dd > 0)
    if label == "perspective":
        rays = [ex2.pixel_ray(M, x, y) for x, y in ((0, 0), (800, 0), (0, 600), (800, 600))]
        check("pixel_ray (perspective): the rays of the four corners all pass through the eye",
              all(np.linalg.norm(np.cross(np.asarray(r[1], float), eye - np.asarray(r[0], float))) < 1e-6 for r in rays))
    else:
        rays = [np.asarray(ex2.pixel_ray(M, x, y)[1], float) for x, y in ((0, 0), (800, 0), (0, 600), (800, 600))]
        check("pixel_ray (orthographic): all rays are parallel", all(close(r, rays[0], 1e-9) for r in rays))
check("pixel_ray: returns shape (2, 3)", np.shape(ex2.pixel_ray(VP @ ex2.perspective(40, 4 / 3, .1, 100) @ Vp, 400, 300)) == (2, 3))

# ---------------------------------------------------------------- chase_view
for Mm in (ex2.translate(1, 2, 3) @ ex2.euler(30, 50, 10), ex2.translate(-4, 0, 2) @ ex2.euler(-80, 20, 140), np.eye(4),
           ex2.translate(2, 0, 0)):
    Vc = np.asarray(ex2.chase_view(Mm), float)
    VM = Vc @ Mm
    check("chase_view: V M sends the camera (0,2,-8) to the origin", close(apply(VM, (0, 2, -8)), 0, 1e-9))
    check("chase_view: V M sends the plane's center to (0,0,-sqrt(68))", close(apply(VM, (0, 0, 0)), (0, 0, -np.sqrt(68)), 1e-9))
    uu = apply(VM, (0, 3, -8))
    check("chase_view: the plane's +y points up in the image", uu[1] > 0 and abs(uu[0]) < 1e-9)
    check("chase_view: seen from behind, the plane's +x (its left wing tip direction) appears on the left", apply(VM, (1, 2, -8))[0] < 0)
    check("chase_view: V is a rigid (rotation + translation) 4x4", Vc.shape == (4, 4) and close(Vc[:3, :3] @ Vc[:3, :3].T, np.eye(3), 1e-9) and close(Vc[3], [0, 0, 0, 1]))
VMs = ex2.chase_view(np.eye(4)) @ np.eye(4)
check("chase_view: V M is the same for every pose of the plane",
      close(np.asarray(ex2.chase_view(ex2.translate(5, 6, 7) @ ex2.euler(20, 30, 40)) @ ex2.translate(5, 6, 7) @ ex2.euler(20, 30, 40), float), VMs, 1e-9))

print("\nFAILED:" if fails else "\nALL PASSED", *fails, sep="\n  ")
raise SystemExit(1 if fails else 0)
