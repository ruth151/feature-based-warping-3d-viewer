"""Triangle meshes for the Exercise 2 viewer: a low-poly airplane, a cube, a square, and a
loader for OBJ files.

A mesh has vertices (Nx3), triangles (Fx3 vertex indices, counterclockwise seen from outside)
and one RGB color per triangle. Back faces are culled unless cull is False (for open surfaces
such as the square).
"""
import numpy as np

GRAY = (0.714, 0.749, 0.800)      # the gray-blue of the course figures, #B6BFCC
RED = (0.85, 0.30, 0.30)
GREEN = (0.30, 0.68, 0.36)
DARK = (0.45, 0.50, 0.58)


class Mesh:
    def __init__(self, verts, faces, colors=None, cull=True):
        self.verts = np.asarray(verts, float)
        self.faces = np.asarray(faces, int)
        if colors is None:
            colors = [GRAY] * len(self.faces)
        self.colors = np.broadcast_to(np.asarray(colors, float), (len(self.faces), 3)).copy()
        self.cull = cull

    @staticmethod
    def join(meshes):
        verts, faces, colors, n = [], [], [], 0
        for m in meshes:
            verts.append(m.verts)
            faces.append(m.faces + n)
            colors.append(m.colors)
            n += len(m.verts)
        return Mesh(np.vstack(verts), np.vstack(faces), np.vstack(colors))


def _convex(verts, faces, color):
    """A convex part: orient every triangle so that its normal points away from the centroid."""
    v = np.asarray(verts, float)
    c = v.mean(0)
    out = []
    for f in faces:
        a, b, d = v[list(f)]
        if np.cross(b - a, d - a) @ ((a + b + d) / 3 - c) < 0:
            f = (f[0], f[2], f[1])
        out.append(f)
    return Mesh(v, out, [color] * len(out))


def _hexahedron(bottom, top, color):
    """A convex solid from two quads (4x3 each, corresponding corners)."""
    v = np.vstack([bottom, top])
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return _convex(v, tris, color)


def _slab(quad, axis, thickness, color):
    """A flat convex quad (4x3) thickened along a coordinate axis."""
    q = np.asarray(quad, float)
    d = np.zeros(3)
    d[axis] = thickness / 2
    return _hexahedron(q - d, q + d, color)


def airplane():
    """A low-poly airplane: nose along +z, up along +y, wings along x. The tip of the left
    wing (at +x) is red and the right one (at -x) green, as on real aircraft."""
    # fuselage: a convex body of revolution, octagonal rings between the tail and the nose
    rings = [(-1.8, 0.18), (-0.5, 0.32), (1.0, 0.32), (1.6, 0.22)]
    k = np.radians(22.5 + 45 * np.arange(8))
    v = [(0, 0.12, -2.2)]
    for z, r in rings:
        v += [(r * np.cos(a), r * np.sin(a), z) for a in k]
    v.append((0, 0, 2.1))
    tris = [(0, 1 + (i + 1) % 8, 1 + i) for i in range(8)]
    for j in range(len(rings) - 1):
        a, b = 1 + 8 * j, 1 + 8 * (j + 1)
        for i in range(8):
            i2 = (i + 1) % 8
            tris += [(a + i, a + i2, b + i2), (a + i, b + i2, b + i)]
    last, tip = 1 + 8 * (len(rings) - 1), len(v) - 1
    tris += [(last + i, last + (i + 1) % 8, tip) for i in range(8)]
    parts = [_convex(v, tris, GRAY)]

    # wings, swept back, each with a colored tip
    def wing(s, x0, x1, color):
        def edge(x):                       # leading and trailing edge at span x
            f = (x - 0.25) / (2.4 - 0.25)
            return (0.7 - 0.6 * f, -0.1 - 0.2 * f)
        (l0, t0), (l1, t1) = edge(x0), edge(x1)
        quad = [(s * x0, -0.05, l0), (s * x1, -0.05, l1), (s * x1, -0.05, t1), (s * x0, -0.05, t0)]
        return _slab(quad, 1, 0.07, color)
    for s, tipcolor in ((1, RED), (-1, GREEN)):
        parts += [wing(s, 0.25, 2.0, GRAY), wing(s, 2.0, 2.4, tipcolor)]

    # horizontal tail
    for s in (1, -1):
        quad = [(s * 0.15, 0.08, -1.55), (s * 0.95, 0.08, -1.85), (s * 0.95, 0.08, -2.1), (s * 0.15, 0.08, -2.0)]
        parts.append(_slab(quad, 1, 0.05, GRAY))
    # vertical fin, darker so that "up" reads at a glance
    quad = [(0, 0.15, -1.4), (0, 1.0, -1.9), (0, 1.0, -2.15), (0, 0.15, -2.1)]
    parts.append(_slab(quad, 0, 0.06, DARK))
    return Mesh.join(parts)


def cube(size=1.0):
    """A cube centered at the origin; the faces normal to x, y and z are tinted red, green and blue."""
    h = size / 2
    v = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
    quads = {(0, 1, 3, 2): (0, 0), (4, 6, 7, 5): (0, 1), (0, 4, 5, 1): (1, 0),
             (2, 3, 7, 6): (1, 1), (0, 2, 6, 4): (2, 0), (1, 5, 7, 3): (2, 1)}
    tint = [(0.85, 0.62, 0.62), (0.62, 0.80, 0.64), (0.62, 0.68, 0.88)]
    tris, colors = [], []
    for (a, b, c, d), (axis, _) in quads.items():
        tris += [(a, b, c), (a, c, d)]
        colors += [tint[axis]] * 2
    m = _convex(v, tris, GRAY)
    m.colors[:] = colors
    return m


def square(size=1.0, color=GRAY):
    """A square in the plane z = 0, centered at the origin, visible from both sides."""
    h = size / 2
    return Mesh([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [(0, 1, 2), (0, 2, 3)],
                [color] * 2, cull=False)


def load_obj(path, size=4.0, color=GRAY):
    """Read the vertices and faces of an OBJ file (polygons are split into triangle fans).
    The mesh is centered at the origin and scaled so that its largest extent is size."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                verts.append([float(t) for t in p[1:4]])
            elif p[0] == 'f':
                idx = [int(t.split('/')[0]) for t in p[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                faces += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    v = np.array(verts)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) * (size / (hi - lo).max())
    return Mesh(v, faces, [color] * len(faces))
