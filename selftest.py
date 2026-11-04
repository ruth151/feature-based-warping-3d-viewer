"""Quick checks that your Part B functions follow the conventions the viewer expects.

    python3 selftest.py             tests ex2.py
    python3 selftest.py mymodule    tests mymodule.py

Passing these checks does not mean your functions are correct: the grading tests are different
and more thorough. A failure here means the viewer will not work as described.
"""
import importlib
import sys
import traceback

import numpy as np


def Rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def Ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def Rz(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def apply(M, p):
    h = np.asarray(M, float) @ np.r_[p, 1.0]
    return h[:3] / h[3]


def close(a, b, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
        raise AssertionError(f"{what}: expected\n{np.round(b, 4)}\ngot\n{np.round(a, 4)}")


def tests(m):
    rng = np.random.default_rng(1)
    k = rng.normal(size=3)
    k /= np.linalg.norm(k)

    def t_translate():
        close(apply(m.translate(1, 2, 3), (1, 1, 1)), (2, 3, 4), 'translate(1, 2, 3) of (1, 1, 1)')

    def t_scale():
        close(apply(m.scale(2, 3, 4), (1, 1, 1)), (2, 3, 4), 'scale(2, 3, 4) of (1, 1, 1)')

    def t_rotate():
        close(apply(m.rotate((0, 0, 1), 90), (1, 0, 0)), (0, 1, 0), 'rotate(z, 90) of (1, 0, 0)')
        close(m.rotate((1, 0, 0), 30), Rx(30), 'rotate(x, 30)')
        close(apply(m.rotate(k, 70), k), k, 'rotate(k, 70) of its own axis k')
        R = np.asarray(m.rotate(k, 70))[:3, :3]
        close(R.T @ R, np.eye(3), 'R^T R for rotate(k, 70)')

    def t_euler():
        close(m.euler(10, 20, 30), Rz(30) @ Ry(20) @ Rx(10), 'euler(10, 20, 30) = Rz(30) Ry(20) Rx(10)')

    def t_quat():
        q = np.asarray(m.quat((0, 0, 1), 90))
        close(q, (np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)), 'quat(z, 90) as [w, x, y, z]')

    def t_quat_to_matrix():
        close(m.quat_to_matrix(m.quat((0, 0, 1), 90)), Rz(90), 'quat_to_matrix(quat(z, 90))')
        close(m.quat_to_matrix(m.quat(k, 70)), m.rotate(k, 70), 'quat_to_matrix(quat(k, 70)) vs rotate(k, 70)')

    def t_quat_mul():
        i, j = np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])
        close(m.quat_mul(i, j), (0, 0, 0, 1), 'i j = k')
        close(m.quat_mul(j, i), (0, 0, 0, -1), 'j i = -k')
        q1, q2 = m.quat((1, 0, 0), 40), m.quat(k, 70)
        close(m.quat_to_matrix(m.quat_mul(q1, q2)), Rx(40) @ m.rotate(k, 70),
              'quat_mul(q1, q2) is q2 followed by q1')

    def t_slerp():
        q0, q1 = m.quat((0, 1, 0), 0), m.quat((0, 1, 0), 90)
        close(m.slerp(q0, q1, 0), q0, 'slerp at t = 0')
        close(m.slerp(q0, q1, 1), q1, 'slerp at t = 1')
        close(m.slerp(q0, q1, 0.25), m.quat((0, 1, 0), 22.5), 'slerp from 0 to 90 degrees at t = 0.25')
        close(m.slerp(q0, m.quat((0, 0, 1), 270), 0.5), m.quat((0, 0, 1), 135),
              'slerp to quat(z, 270) at t = 0.5: the arc to q1 as given, without replacing q1 by -q1')

    def t_look_at():
        V = m.look_at((3, 2, 5), (1, 0, 1), (0, 1, 0))
        close(apply(V, (3, 2, 5)), (0, 0, 0), 'look_at: the eye goes to the origin')
        d = np.linalg.norm(np.subtract((3, 2, 5), (1, 0, 1)))
        close(apply(V, (1, 0, 1)), (0, 0, -d), 'look_at: the target goes to the -z axis')
        u = apply(V, (3, 3, 5))                        # a point above the eye
        if u[1] <= 0 or abs(u[0]) > 1e-9:
            raise AssertionError(f'look_at: a point above the eye should have x = 0 and y > 0, got {u}')
        close(np.asarray(V)[:3, :3] @ np.asarray(V)[:3, :3].T, np.eye(3), 'look_at: the rotation part is orthonormal')

    def t_perspective():
        P = m.perspective(90, 2, 1, 10)
        close(apply(P, (0, 0, -1)), (0, 0, -1), 'perspective: the near plane goes to z = -1')
        close(apply(P, (0, 0, -10)), (0, 0, 1), 'perspective: the far plane goes to z = 1')
        close(apply(P, (2, 1, -1))[:2], (1, 1), 'perspective(90, aspect 2): the top right corner of the near plane')

    def t_ortho():
        P = m.ortho(-2, 4, -1, 3, 1, 10)
        close(apply(P, (-2, -1, -1)), (-1, -1, -1), 'ortho: corner (l, b, -near)')
        close(apply(P, (4, 3, -10)), (1, 1, 1), 'ortho: corner (r, t, -far)')

    def t_viewport():
        VP = m.viewport(800, 600)
        close(apply(VP, (-1, 1, -1)), (0, 0, 0), 'viewport: NDC (-1, 1, -1) is the top left pixel corner, depth 0')
        close(apply(VP, (1, -1, 1)), (800, 600, 1), 'viewport: NDC (1, -1, 1) is the bottom right corner, depth 1')

    def t_project():
        M = np.diag([1.0, 1, 1, 2])
        close(m.project(M, np.array([[2.0, 4, 6], [1, 1, 1]])), [[1, 2, 3], [0.5, 0.5, 0.5]],
              'project: divide by w')

    def t_chase_view():
        Ms = [m.translate(1, 2, 3) @ m.euler(30, 50, 10), m.translate(-4, 0, 2) @ m.euler(-80, 20, 140)]
        for M in Ms:
            VM = np.asarray(m.chase_view(M), float) @ np.asarray(M, float)
            close(apply(VM, (0, 2, -8)), (0, 0, 0), 'chase_view: the camera, at (0, 2, -8) of the airplane, goes to the origin')
            close(apply(VM, (0, 0, 0)), (0, 0, -np.sqrt(68)), "chase_view: the airplane's center goes to the -z axis")
            u = apply(VM, (0, 3, -8))                  # a point above the camera, in the airplane's frame
            if u[1] <= 0 or abs(u[0]) > 1e-6:
                raise AssertionError(f"chase_view: the airplane's +y should point up in the image, got {u}")

    def t_pixel_ray():
        M = np.asarray(m.viewport(800, 600) @ m.perspective(40, 4 / 3, 0.1, 100) @ m.look_at((6, 4, 8), (0, 0, 0), (0, 1, 0)))
        r = m.pixel_ray(M, 400, 300)
        if np.shape(r) != (2, 3):
            raise AssertionError(f'pixel_ray should return (o, d), two 3-vectors; got shape {np.shape(r)}')
        o, d = np.asarray(r, float)
        close(apply(M, o), (400, 300, 0), 'pixel_ray: o projects to (400, 300) at depth 0')
        close(d, -np.array([6, 4, 8]) / np.linalg.norm([6, 4, 8]), 'pixel_ray: the center pixel looks along the viewing direction')
        o, d = np.asarray(m.pixel_ray(M, 100, 50), float)
        close(np.linalg.norm(d), 1.0, 'pixel_ray: d is a unit vector')
        close(apply(M, o + 5 * d)[:2], (100, 50), 'pixel_ray: every point o + s d projects to the same pixel')

    return [(name[2:], f) for name, f in locals().items() if name.startswith('t_')]


class Checked:
    """The module's functions, each raising a clear error when it returns None (a stub)."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        f = getattr(self.module, name)

        def g(*args):
            r = f(*args)
            if r is None:
                raise AssertionError(f'{name} returned None (not implemented yet?)')
            return r
        return g


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'ex2'
    module = importlib.import_module(name)
    m = Checked(module)
    failed = 0
    for fname, test in tests(m):
        try:
            if getattr(module, fname, None) is None:
                raise AssertionError('missing')
            test()
            print(f'  ok    {fname}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fname}: {e}')
        except Exception as e:
            failed += 1
            last = traceback.extract_tb(e.__traceback__)[-1]
            print(f'  FAIL  {fname}: {type(e).__name__}: {e}  (line {last.lineno} of {last.filename.split("/")[-1]})')
    print('all checks passed' if not failed else f'{failed} of {len(tests(m))} functions failed')


if __name__ == '__main__':
    main()
