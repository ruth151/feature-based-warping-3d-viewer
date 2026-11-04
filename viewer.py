"""Exercise 2 viewer: a small mesh viewer built on your Part B functions.

Every matrix the viewer uses comes from your ex2.py: the rotation of the object (euler, or the
quaternion functions), the view matrix (look_at, or chase_view for the chase camera), the
projection (perspective or ortho), the viewport, and project, which takes the vertices to raster
coordinates. A click moves the airplane along the ray from your pixel_ray. If one of them is
wrong, you will see it. The viewer itself only draws: it shades each triangle by how directly it faces
the camera, skips the triangles facing away, and paints the rest from back to front.

Interactive:
    python3 viewer.py                     the airplane
    python3 viewer.py --mesh bunny.obj    any OBJ mesh
    python3 viewer.py --scene squares     two equal squares at depths 0 and -5 (dolly zoom)
    python3 viewer.py --key0 0,0,0 --key1 180,180,180    keyframes for the interpolation

From your own code (for the report figures):
    import viewer
    scene = viewer.Scene()
    scene.add(viewer.airplane(), ex2.euler(0, 90, 0))
    scene.world_axes = 3.0                # draw the world axes X, Y, Z (optional)
    cam = viewer.Camera(eye=(6, 4, 8), fovy=40)
    viewer.render(scene, cam, "fig.png")
"""
import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshes import Mesh, airplane, cube, square, load_obj, GRAY  # noqa: E402,F401

FUNCTIONS = ['translate', 'scale', 'rotate', 'euler', 'quat', 'quat_mul', 'quat_to_matrix',
             'slerp', 'look_at', 'ortho', 'perspective', 'viewport', 'project', 'chase_view', 'pixel_ray']
CHASE_EYE = (0.0, 2.0, -8.0)               # the chase camera, in the airplane's frame
GRID = 6                                   # the ground grid covers -GRID <= x, z <= GRID

_impl = None


def use(module):
    """Choose the module that provides the functions: a module, or its name (default 'ex2')."""
    global _impl
    _impl = importlib.import_module(module) if isinstance(module, str) else module


def call(name, *args):
    """Call one of your functions, with a clear message if it is missing or returns None."""
    if _impl is None:
        use('ex2')
    f = getattr(_impl, name, None)
    if f is None:
        raise NotImplementedError(f"{_impl.__name__}.{name} is missing")
    r = f(*args)
    if r is None:
        raise NotImplementedError(f"{_impl.__name__}.{name} returned None (not implemented yet?)")
    return np.asarray(r, float)


# --- scene and camera ------------------------------------------------------------------------

class Scene:
    def __init__(self):
        self.items = []                    # (mesh, 4x4 model matrix)
        self.labels = []                   # (text, 3D world point)
        self.world_axes = 0.0              # length of the world axes X, Y, Z to draw (0: none)
        self.object_axes = []              # (4x4 model matrix, length): the object's axes x, y, z
        self.bold = None                   # 'world' or 'object': the axes to draw bold
        self.grid = None                   # height of a ground grid (y = const), or None

    def add(self, mesh, M=None):
        self.items.append((mesh, np.eye(4) if M is None else np.asarray(M, float)))
        return self


class Camera:
    def __init__(self, eye=(6, 4, 8), target=(0, 0, 0), up=(0, 1, 0), fovy=40.0, near=0.1,
                 far=100.0, ortho=False, height=None, view=None):
        """For an orthographic camera, height is the height of the view volume; by default it is
        the height that the perspective camera sees at the target, so switching keeps the size.
        view: a view matrix to use instead of look_at(eye, target, up)."""
        self.eye, self.target, self.up = (np.asarray(a, float) for a in (eye, target, up))
        self.fovy, self.near, self.far, self.ortho, self.height = fovy, near, far, ortho, height
        self.view = view

    def projection(self, aspect):
        if not self.ortho:
            return call('perspective', self.fovy, aspect, self.near, self.far)
        h = self.height
        if h is None:
            h = 2 * np.linalg.norm(self.eye - self.target) * np.tan(np.radians(self.fovy) / 2)
        return call('ortho', -h * aspect / 2, h * aspect / 2, -h / 2, h / 2, self.near, self.far)

    def describe(self, aspect):
        if not self.ortho:
            return f'perspective({self.fovy:.1f}, {aspect:.3f}, {self.near:g}, {self.far:g})'
        h = self.height or 2 * np.linalg.norm(self.eye - self.target) * np.tan(np.radians(self.fovy) / 2)
        w = h * aspect
        return f'ortho({-w / 2:.2f}, {w / 2:.2f}, {-h / 2:.2f}, {h / 2:.2f}, {self.near:g}, {self.far:g})'

    def matrices(self, W, H):
        V = self.view if self.view is not None else call('look_at', self.eye, self.target, self.up)
        return V, self.projection(W / H), call('viewport', W, H)


# --- drawing ---------------------------------------------------------------------------------

AXIS_COLORS = ['#d03030', '#20a040', '#3060d0']
EDGE = '#30343a'                           # the outlines of the triangles
LIGHT = np.array([-0.4, 0.7, 1.0]) / np.linalg.norm([-0.4, 0.7, 1.0])   # eye coordinates: above left of the camera


def _homog(M, pts):
    """M applied to points, without the division by w (the viewer's own product)."""
    return (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]


def build(scene, cam, W, H):
    """Everything to draw, in raster coordinates: polygons with colors (back to front), lines
    (grid, axes) and labels."""
    V, P, VP = cam.matrices(W, H)
    polys, colors, depths = [], [], []
    for mesh, M in scene.items:
        MV = V @ M
        eye = _homog(MV, mesh.verts)                     # eye coordinates: shading and sorting
        ras = call('project', VP @ P @ MV, mesh.verts)   # your projection: what is drawn
        tri = eye[mesh.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
        c = tri.mean(1)
        view = np.array([0, 0, 1.0]) if cam.ortho else -c / np.linalg.norm(c, axis=1, keepdims=True)
        facing = (n * view).sum(1)
        keep = (tri[:, :, 2] < -cam.near).all(1)         # in front of the near plane
        if mesh.cull:
            keep &= facing > 0
        diffuse = n @ LIGHT
        shade = 0.4 + 0.6 * (np.abs(diffuse) if not mesh.cull else np.maximum(diffuse, 0))
        for i in np.flatnonzero(keep):
            polys.append(ras[mesh.faces[i], :2])
            colors.append(np.clip(mesh.colors[i] * shade[i], 0, 1))
            depths.append(c[i, 2])
    order = np.argsort(depths)                           # most negative z (farthest) first
    polys = [polys[i] for i in order]
    colors = [colors[i] for i in order]

    lines = []                                           # (2x2 raster points, color, width, zorder)
    M_all = VP @ P @ V

    def segment(a, b, color, width, z):
        e = _homog(V, np.array([a, b]))
        if (e[:, 2] < -cam.near).all():
            lines.append((call('project', M_all, np.array([a, b]))[:, :2], color, width, z))
    if scene.grid is not None:
        y = scene.grid
        for s in range(-GRID, GRID + 1):
            segment((s, y, -GRID), (s, y, GRID), '#d8d8d8', 0.8, 0)
            segment((-GRID, y, s), (GRID, y, s), '#d8d8d8', 0.8, 0)
    # the world axes X, Y, Z (thin) and the object's axes x, y, z (thicker), or the bold ones
    # drawn last, on top
    frames = []
    if scene.world_axes:
        frames.append(('world', np.eye(4), scene.world_axes, 'XYZ', 1.2))
    frames += [('object', M, L, 'xyz', 2.2) for M, L in scene.object_axes]
    frames.sort(key=lambda fr: fr[0] == scene.bold)
    names = [(text, X, 'black') for text, X in scene.labels]
    for which, M, L, letters, width in frames:
        if scene.bold:
            width = 3.2 if which == scene.bold else 1.0
        o = M[:3, 3]
        for k in range(3):
            d = M[:3, k] / np.linalg.norm(M[:3, k])
            segment(o, o + L * d, AXIS_COLORS[k], width, 3)
            names.append((letters[k], o + (L + 0.3) * d, AXIS_COLORS[k]))
    labels = []
    for text, X, color in names:
        if (_homog(V, np.array([X]))[:, 2] < -cam.near).all():
            labels.append((text, call('project', M_all, np.array([X]))[0, :2], color))
    return polys, colors, lines, labels


def _setup_axes(ax, W, H):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                                    # raster y points down
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#c0c0c0')


class View:
    """The artists of one W x H view, created once and updated in place (fast enough to drag)."""

    def __init__(self, ax, W=800, H=600, animated=False):
        """animated: the artists are left out of ordinary figure draws, and drawn by blitting."""
        from matplotlib.collections import PolyCollection, LineCollection
        self.ax, self.W, self.H, self.animated = ax, W, H, animated
        _setup_axes(ax, W, H)
        self.back = LineCollection([], zorder=0, animated=animated)    # the ground grid
        self.polys = PolyCollection([], edgecolors=EDGE, linewidths=0.25, zorder=1,
                                    animated=animated)
        self.front = LineCollection([], zorder=3, animated=animated)   # the axes, over the mesh
        for c in (self.back, self.polys, self.front):
            ax.add_collection(c)
        self.labels = []
        self.message = ax.text(W / 2, H / 2, '', ha='center', va='center', fontsize=12, wrap=True,
                               animated=animated)

    def artists(self):
        return [self.back, self.polys, self.front, *self.labels, self.message]

    def show(self, scene, cam):
        polys, colors, lines, labels = build(scene, cam, self.W, self.H)
        self.polys.set_verts(polys)
        self.polys.set_facecolor(colors if colors else 'none')
        # outline the triangles of a coarse mesh; in a dense one the outlines would darken it
        self.polys.set_edgecolor(EDGE if len(polys) < 1000 else colors or 'none')
        for coll, front in ((self.back, False), (self.front, True)):
            sel = [l for l in lines if (l[3] > 0) == front]
            coll.set_segments([l[0] for l in sel])
            coll.set_color([l[1] for l in sel] or 'none')
            coll.set_linewidth([l[2] for l in sel] or 1)
        while len(self.labels) < len(labels):
            self.labels.append(self.ax.text(0, 0, '', ha='center', va='center', fontsize=12, zorder=4,
                                            animated=self.animated, clip_on=True))
        for i, t in enumerate(self.labels):
            t.set_text(labels[i][0] if i < len(labels) else '')
            if i < len(labels):
                t.set_position(labels[i][1])
                t.set_color(labels[i][2])
        self.message.set_text('')

    def show_message(self, text):
        self.polys.set_verts([])
        self.back.set_segments([])
        self.front.set_segments([])
        for t in self.labels:
            t.set_text('')
        self.message.set_text(text)


def draw(ax, scene, cam, W=800, H=600):
    View(ax, W, H).show(scene, cam)


def render(scene, cam, file=None, W=800, H=600, title=None):
    """Draw the scene as the camera sees it, in a W x H image; save it if file is given."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W / 100, H / 100 + (0.4 if title else 0)))
    ax = fig.add_axes([0, 0, 1, H / (H + (40 if title else 0))])
    draw(ax, scene, cam, W, H)
    if title:
        fig.suptitle(title, y=0.99)
    if file:
        fig.savefig(file, dpi=150)
        plt.close(fig)
    return fig


def render_strip(scenes, cam, file=None, W=400, H=300, labels=None):
    """Draw several scenes side by side (for example the frames of an interpolation)."""
    import matplotlib.pyplot as plt
    n = len(scenes)
    fig, axs = plt.subplots(1, n, figsize=(n * W / 100, H / 100 + 0.4), squeeze=False)
    for i, (ax, s) in enumerate(zip(axs[0], scenes)):
        draw(ax, s, cam, W, H)
        if labels:
            ax.set_title(labels[i], fontsize=11)
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=150)
        plt.close(fig)
    return fig


# --- rotations -------------------------------------------------------------------------------

def euler_to_quat(ax, ay, az):
    """The quaternion of euler(ax, ay, az), from your quaternion functions."""
    return call('quat_mul', call('quat', (0, 0, 1), az),
                call('quat_mul', call('quat', (0, 1, 0), ay), call('quat', (1, 0, 0), ax)))


def _euler_angles(R):
    """Angles (ax, ay, az) with euler(ax, ay, az) = R; at gimbal lock (ay = +-90) ax = 0."""
    sy = np.clip(-R[2, 0], -1, 1)
    ay = np.degrees(np.arcsin(sy))
    if abs(sy) < 1 - 1e-9:
        ax = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        az = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        ax, az = 0.0, np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
    return np.array([ax, ay, az])


# --- the interactive viewer ------------------------------------------------------------------

MODES = ['Euler angles', 'world axes', 'object axes']
TURN_TIME = 0.25                           # seconds per turn of the x, y, z keys
METHODS = ['Euler angles', 'slerp', 'side by side']
HELP = """keys   x y z turn by the step (shift: back)   e w o key mode
       p orthographic   c chase camera   space play   r reset
mouse  click: move the airplane to the point of y = 0 under it
       drag: orbit the camera   scroll: move closer"""


class App:
    W, H = 800, 600

    def __init__(self, mesh=None, scene='airplane', key0=(0, 0, 0), key1=(0, 0, 0)):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
        for k in list(matplotlib.rcParams):
            if k.startswith('keymap.') and k != 'keymap.quit':
                matplotlib.rcParams[k] = []
        matplotlib.rcParams['toolbar'] = 'None'

        self.squares = scene == 'squares'
        self.mesh = mesh if mesh is not None else airplane()
        self.angles = np.zeros(3)          # Euler angles (the sliders)
        self.q = np.array([1.0, 0, 0, 0])  # the orientation, in the world / object axes modes
        self.mode = 0
        self.keys = [self._key(key0), self._key(key1)]
        self.t = None                      # interpolation parameter on display, or None
        self.method = 0
        self.timer = None                  # plays the interpolation
        self.turning = None                # the turn in progress (animated)
        self.turn_timer = None
        self.error = None
        self.cam = dict(az=0.0, el=0.0, dist=5.0, fovy=40.0) if self.squares else \
            dict(az=35.0, el=20.0, dist=10.0, fovy=40.0)
        self.ortho = False
        self.chase = False                 # look through the chase camera (chase_view)
        self.pos = np.zeros(3)             # the airplane's center, moved by a click
        self.click = None                  # the last click: pixel, ray and the point reached
        self.drag = None
        self._syncing = False

        self.fig = plt.figure(figsize=(14, 8.4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('Exercise 2 viewer')
        # The view and the text panel are drawn by blitting: while you drag, only the view is
        # redrawn, and the text (slow to draw) at most a few times a second.
        self.ax = self.fig.add_axes([0.01, 0.29, 0.6, 0.69])
        self.view = View(self.ax, self.W, self.H, animated=True)
        self.info_ax = self.fig.add_axes([0.63, 0.29, 0.36, 0.69])
        self.info_ax.axis('off')
        self.info = self.info_ax.text(0, 1, '', va='top', family='monospace', fontsize=9.5,
                                      animated=True)
        self._bg = self._bg_info = self._info_img = None
        self._info_key = None              # (text, bounds) of the cached text panel
        self._info_time = 0.0
        self._info_timer = None
        self._quiet = False

        def slider(rect, label, lo, hi, v, step=None):
            s = Slider(self.fig.add_axes(rect), label, lo, hi, valinit=v, valstep=step)
            return s
        self.s_ang = [slider([0.08, 0.215 - 0.04 * i, 0.3, 0.025], f'a{"xyz"[i]}', -180, 180, 0, 1)
                      for i in range(3)]
        self.s_t = slider([0.08, 0.06, 0.3, 0.025], 'interp t', 0, 1, 0, 0.01)
        self.s_fovy = slider([0.5, 0.215, 0.18, 0.025], 'fovy', 1, 120, self.cam['fovy'], 0.1)
        self.s_dist = slider([0.5, 0.175, 0.18, 0.025], 'distance', 1, 60, self.cam['dist'], 0.1)
        self.s_step = slider([0.5, 0.135, 0.18, 0.025], 'key step', 1, 90, 15, 1)
        for i, s in enumerate(self.s_ang):
            s.on_changed(lambda v, i=i: self._on_angle(i, v))
        self.s_t.on_changed(self._on_t)
        self.s_fovy.on_changed(lambda v: self._on_cam('fovy', v))
        self.s_dist.on_changed(lambda v: self._on_cam('dist', v))

        self.fig.text(0.745, 0.245, 'rotation keys', fontsize=10)
        self.r_mode = RadioButtons(self.fig.add_axes([0.74, 0.13, 0.11, 0.11], frameon=False), MODES)
        self.r_mode.on_clicked(lambda label: self.set_mode(MODES.index(label)))
        self.fig.text(0.865, 0.245, 'interpolation', fontsize=10)
        self.r_method = RadioButtons(self.fig.add_axes([0.86, 0.13, 0.12, 0.11], frameon=False), METHODS)
        self.r_method.on_clicked(self._on_method)
        self.b = {}
        for i, (name, fn) in enumerate([('set key 0', lambda e: self.set_key(0)),
                                        ('set key 1', lambda e: self.set_key(1)),
                                        ('play', lambda e: self.play()),
                                        ('reset', lambda e: self.reset())]):
            self.b[name] = Button(self.fig.add_axes([0.5 + 0.075 * i, 0.05, 0.065, 0.04]), name)
            self.b[name].on_clicked(fn)
        self.c_opts = CheckButtons(self.fig.add_axes([0.81, 0.0, 0.18, 0.125], frameon=False),
                                   ['orthographic', 'chase camera', 'axes', 'ground grid'],
                                   [False, False, not self.squares, not self.squares])
        self.c_opts.on_clicked(self._on_option)

        c = self.fig.canvas
        c.mpl_connect('key_press_event', self._on_key)
        c.mpl_connect('button_press_event', self._on_press)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('motion_notify_event', self._on_motion)
        c.mpl_connect('scroll_event', self._on_scroll)
        c.mpl_connect('draw_event', self._on_draw)
        self.update()

    # state ---------------------------------------------------------------------------------
    def _key(self, angles):
        a = np.asarray(angles, float)
        try:
            q = euler_to_quat(*a)
        except NotImplementedError:
            q = None
        return dict(angles=a, q=q)

    def rotation(self):
        """The 4x4 rotation of the object now, from your functions."""
        if self.mode == 0:
            return call('euler', *self.angles)
        return call('quat_to_matrix', self.q)

    def model(self):
        """The airplane's model matrix: its rotation (or the interpolated one), then the move to
        its position."""
        R = self.rotation() if self.t is None else self.interpolated(self.method, self.t)
        if not self.pos.any():
            return R
        return call('translate', *self.pos) @ R

    def interpolated(self, method, t):
        k0, k1 = self.keys
        if method == 0:
            return call('euler', *((1 - t) * k0['angles'] + t * k1['angles']))
        if k0['q'] is None or k1['q'] is None:
            raise NotImplementedError('the keys need quat and quat_mul')
        return call('quat_to_matrix', call('slerp', k0['q'], k1['q'], t))

    def _sync_sliders(self):
        self._syncing = True
        for s, v in zip(self.s_ang, self.angles):
            s.set_val(float(np.round(v, 1)) + 0.0)
        self._syncing = False

    def set_mode(self, m):
        self.finish_turn()
        if m == self.mode:
            return
        try:
            if self.mode == 0:                    # Euler angles -> quaternion
                self.q = euler_to_quat(*self.angles)
            self.mode = m
            if m == 0:                            # back to the sliders, keeping the orientation
                self.angles = _euler_angles(call('quat_to_matrix', self.q))
                self._sync_sliders()
        except NotImplementedError as e:
            self.error = str(e)
        if self.r_mode.value_selected != MODES[self.mode]:
            self.r_mode.set_active(self.mode)
        self.t = None
        self.update()

    def turn(self, axis, sign):
        """Rotate by one step about the x, y or z axis (0, 1, 2) in the current mode. The turn
        is animated, so that you can see which axis the object turns about."""
        import time
        self.finish_turn()
        self.stop_play()
        self.t = None
        self.turning = dict(axis=axis, step=sign * self.s_step.val, start=time.monotonic(),
                            angles=self.angles.copy(), q=self.q.copy())
        self.turn_timer = self.fig.canvas.new_timer(interval=15)
        self.turn_timer.add_callback(self._turn_tick)
        self.turn_timer.start()
        self._turn_tick()

    def _turn_tick(self):
        import time
        if self.turning is None:
            return
        f = (time.monotonic() - self.turning['start']) / TURN_TIME
        if f >= 1:
            return self.finish_turn()
        try:
            self._apply_turn(f)
        except NotImplementedError as e:
            self.error = str(e)
            return self.finish_turn()
        self.update(fast=True)

    def _apply_turn(self, f):
        """Set the orientation reached after the fraction f of the turn in progress."""
        a = self.turning
        if self.mode == 0:
            v = a['angles'][a['axis']] + f * a['step']
            if f == 1:
                v = v - 360 if v > 180 else v + 360 if v < -180 else v
            self.angles[a['axis']] = v
        else:
            dq = call('quat', np.eye(3)[a['axis']], f * a['step'])
            q = call('quat_mul', dq, a['q']) if self.mode == 1 else call('quat_mul', a['q'], dq)
            self.q = q / np.linalg.norm(q)
            self.angles = _euler_angles(call('quat_to_matrix', self.q))

    def finish_turn(self):
        """End the turn in progress, if any, at its final orientation."""
        if self.turning is None:
            return
        self.turn_timer.stop()
        try:
            self._apply_turn(1.0)
        except NotImplementedError as e:
            self.error = str(e)
        self.turning = None
        self._sync_sliders()
        self.update()

    def set_key(self, i):
        self.finish_turn()
        try:
            q = euler_to_quat(*self.angles) if self.mode == 0 else self.q.copy()
        except NotImplementedError:
            q = None
        self.keys[i] = dict(angles=self.angles.copy(), q=q)
        self.update()

    def stop_play(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def play(self):
        self.finish_turn()
        self.stop_play()
        self.t = 0.0
        self.timer = self.fig.canvas.new_timer(interval=30)

        def tick():
            self.t = min(1.0, self.t + 1 / 70)
            if self.t < 1:
                self.update(fast=True)
            else:
                self.stop_play()
                self._syncing = True
                self.s_t.set_val(self.t)
                self._syncing = False
                self.update()
        self.timer.add_callback(tick)
        self.timer.start()

    def reset(self):
        self.finish_turn()
        self.stop_play()
        self.angles[:] = 0
        self.q = np.array([1.0, 0, 0, 0])
        self.pos = np.zeros(3)
        self.click = None
        self.t = None
        self._sync_sliders()
        self.update()

    # events --------------------------------------------------------------------------------
    def _on_angle(self, i, v):
        if self._syncing:
            return
        self.finish_turn()
        self.stop_play()
        if self.mode != 0:
            self.mode = 0
            self.r_mode.set_active(0)
        self.angles[i] = v
        self.t = None
        self.update()

    def _on_t(self, v):
        if not self._syncing:
            self.t = v
            self.update()

    def _on_cam(self, k, v):
        self.cam[k] = v
        self.update()

    def _on_method(self, label):
        self.method = METHODS.index(label)
        self.update()

    def _on_option(self, label):
        if label == 'orthographic':
            self.ortho = not self.ortho
        elif label == 'chase camera':
            self.chase = not self.chase
        self.update()

    def _on_key(self, e):
        k = e.key or ''
        if k in ('x', 'y', 'z', 'X', 'Y', 'Z', 'shift+x', 'shift+y', 'shift+z'):
            c = k[-1]
            self.turn('xyz'.index(c.lower()), -1 if (c.isupper() or k.startswith('shift')) else 1)
        elif k in ('e', 'w', 'o'):
            self.set_mode('ewo'.index(k))
        elif k == 'p':
            self.c_opts.set_active(0)
        elif k == 'c':
            self.c_opts.set_active(1)
        elif k == ' ':
            self.play()
        elif k == 'r':
            self.reset()

    def _on_press(self, e):
        if e.inaxes is self.ax and e.button == 1:
            self.drag = dict(x=e.x, y=e.y, az=self.cam['az'], el=self.cam['el'], moved=False,
                             pixel=(e.xdata, e.ydata))

    def _on_motion(self, e):
        d = self.drag
        if d is None:
            return
        if not d['moved']:
            if np.hypot(e.x - d['x'], e.y - d['y']) < 4 * self.fig.dpi / 100:
                return                     # still a click
            d['moved'] = True
        self.cam['az'] = d['az'] - 0.4 * (e.x - d['x'])
        self.cam['el'] = float(np.clip(d['el'] - 0.4 * (e.y - d['y']), -89, 89))
        self.update(fast=True)

    def _on_release(self, e):
        d, self.drag = self.drag, None
        if d is None:
            return
        if not d['moved']:
            self.place(*d['pixel'])
        self.update()

    def place(self, x, y):
        """Move the airplane's center to the point of the plane y = 0 on the ray of the raster
        point (x, y), from your pixel_ray."""
        if self.squares or self.side_by_side():
            return
        try:
            V, P, VP = self.camera().matrices(self.W, self.H)
            r = call('pixel_ray', VP @ P @ V, x, y)
            if r.shape != (2, 3):
                raise NotImplementedError(f'pixel_ray should return (o, d), two 3-vectors; got shape {r.shape}')
        except NotImplementedError as e:
            self.error = str(e)
            return
        o, d = r
        self.click = dict(pixel=(x, y), o=o, d=d, hit=None, moved=False)
        if abs(d[1]) < 1e-12 or -o[1] / d[1] < 0:
            return                         # the ray does not reach the plane y = 0
        hit = o - o[1] / d[1] * d
        hit[1] = 0.0
        self.click['hit'] = hit
        if np.abs(hit[[0, 2]]).max() <= GRID:  # within the ground grid
            self.pos = hit
            self.click['moved'] = True

    def _on_scroll(self, e):
        if e.inaxes is self.ax:
            d = float(np.clip(self.cam['dist'] * 1.1 ** (-e.step), 1, 60))
            self.s_dist.set_val(d)

    # drawing -------------------------------------------------------------------------------
    def camera(self):
        a, el, d = np.radians(self.cam['az']), np.radians(self.cam['el']), self.cam['dist']
        if self.side_by_side():
            d *= 1.8                       # pulled back, to fit both airplanes
        target = np.zeros(3)
        eye = target + d * np.array([np.cos(el) * np.sin(a), np.sin(el), np.cos(el) * np.cos(a)])
        if self.chase and not self.squares and not self.side_by_side():
            V = call('chase_view', self.model())
            if V.shape != (4, 4):
                raise NotImplementedError(f'chase_view should return a 4x4 matrix; got shape {V.shape}')
            h = 2 * np.linalg.norm(CHASE_EYE) * np.tan(np.radians(self.cam['fovy']) / 2)
            return Camera(eye, target, (0, 1, 0), self.cam['fovy'], 0.1, 200.0, self.ortho, h, view=V)
        return Camera(eye, target, (0, 1, 0), self.cam['fovy'], 0.1, 200.0, self.ortho)

    def side_by_side(self):
        return self.t is not None and self.method == 2 and not self.squares

    def scene(self):
        s = Scene()
        show_axes, grid = self.c_opts.get_status()[2:]
        if self.squares:
            # equal squares facing the camera, side by side, at depths 0 and -5 (the camera
            # orbits the origin, so they are at distances d and d + 5 when it looks along -z)
            s.add(square(1.0), call('translate', -0.75, 0, 0))
            s.add(square(1.0, (0.55, 0.62, 0.75)), call('translate', 0.75, 0, -5))
            if grid:
                s.grid = -1.0
            return s
        if grid:
            s.grid = -1.5
        if self.side_by_side():
            for x, m, label in ((-3, 0, 'Euler angles'), (3, 1, 'slerp')):
                M = call('translate', x, 0, 0) @ self.interpolated(m, self.t)
                s.add(self.mesh, M)
                s.labels.append((label, (x, 3.3, 0)))
                if show_axes:
                    s.object_axes.append((M, 2.5))
            return s
        M = self.model()
        s.add(self.mesh, M)
        if show_axes:
            s.world_axes = 3.0
            s.object_axes.append((M, 2.6))
            if self.t is None:
                s.bold = [None, 'world', 'object'][self.mode]
        return s

    def update(self, fast=False):
        """Recompute the view and the text; fast: blit them, without redrawing the figure."""
        lines, n = [], 1
        cam = None
        try:
            cam = self.camera()
            s = self.scene()
            n = len(s.items)
            self.view.show(s, cam)
            V, P, _ = cam.matrices(self.W, self.H)
            R = s.items[0][1] if s.items else np.eye(4)
        except NotImplementedError as e:
            self.view.show_message(str(e))
            V = P = R = None
            self.error = str(e)
        num = lambda v, d=3: f'{np.round(v, d) + 0.0:{d + 4}.{d}f}'     # no "-0.000"
        f = lambda M: '\n'.join('  ' + ' '.join(num(v) for v in row) for row in M)
        vec = lambda q: '[' + ' '.join(num(v) for v in q) + ']'
        if self.t is None:
            lines.append(['keys change the Euler angles',
                          'keys turn about the world axes X Y Z (bold)',
                          'keys turn about the object axes x y z (bold)'][self.mode])
        else:
            lines.append(f'showing: interpolation ({METHODS[self.method]}), t = {self.t:.2f}')
        lines.append('Euler angles  ' + '  '.join(f'{"xyz"[i]} {v + 0.0:.1f}' for i, v in enumerate(self.angles)))
        if self.mode != 0:
            lines.append('quaternion    ' + vec(self.q))
        for i, k in enumerate(self.keys):
            qs = '' if k['q'] is None else '  q ' + vec(k['q'])
            lines.append(f'key {i}: (' + ', '.join(f'{v + 0.0:.0f}' for v in k['angles']) + ')' + qs)
        if R is not None:
            if self.t is not None:
                how = 'interpolated' + (' (left object)' if n > 1 else '')
            else:
                how = 'euler(ax, ay, az)' if self.mode == 0 else 'quat_to_matrix(q)'
            if self.pos.any() and n == 1:
                how = 'translate(' + ', '.join(f'{v + 0.0:.2f}' for v in self.pos) + ') @ ' + how
            lines += ['', 'model matrix M = ' + how, f(R), '']
            if cam.view is not None:
                lines += ['V M,  where V = chase_view(M)', f(V @ R), '']
            else:
                lines += ['view matrix  V = look_at(eye, (0, 0, 0), (0, 1, 0))',
                          '  eye = (' + ', '.join(f'{v + 0.0:.2f}' for v in cam.eye) + ')', f(V), '']
            lines += ['projection  P = ' + cam.describe(self.W / self.H), f(P)]
        if self.click is not None:
            c = self.click
            lines += ['', 'click at (' + ', '.join(f'{v:.1f}' for v in c['pixel']) + '):  pixel_ray gives',
                      '  o = ' + vec(c['o']) + '   d = ' + vec(c['d']),
                      '  the ray does not reach the plane y = 0' if c['hit'] is None else
                      '  center moved to ' + vec(c['hit']) if c['moved'] else
                      '  it meets y = 0 at ' + vec(c['hit']) + ', beyond the grid: no move']
        if self.error:
            lines += ['', 'error: ' + self.error]
            self.error = None
        lines += ['', HELP]
        self.info.set_text('\n'.join(lines))
        if fast and self._bg is not None:
            c = self.fig.canvas
            c.restore_region(self._bg)
            self._paint_view()
            c.blit(self.ax.bbox)
            self._paint_info(blit=True)
        else:
            self.fig.canvas.draw_idle()

    # blitting ------------------------------------------------------------------------------
    def _on_draw(self, e):
        """After every full draw: keep the backgrounds, then draw the view and the text."""
        if self._quiet:
            return
        c = self.fig.canvas
        self._bg = c.copy_from_bbox(self.ax.bbox)
        self._bg_info = c.copy_from_bbox(self.info_ax.bbox)
        self._paint_view()
        self._paint_info(blit=False)

    def _paint_view(self):
        for a in self.view.artists():
            self.ax.draw_artist(a)

    def _paint_info(self, blit):
        """Draw the text panel, or reuse the last drawing if the text is the same or was drawn
        less than 0.25 s ago (then draw it a little later)."""
        import time
        c = self.fig.canvas
        bounds = tuple(np.round(self.info_ax.bbox.bounds))
        key = (self.info.get_text(), bounds)
        cached = self._info_img is not None and self._info_key[1] == bounds
        now = time.monotonic()
        if cached and (self._info_key == key or now - self._info_time < 0.25):
            c.restore_region(self._info_img)
            if self._info_key != key and self._info_timer is None:
                self._info_timer = c.new_timer(interval=300)
                self._info_timer.single_shot = True
                self._info_timer.add_callback(self._info_later)
                self._info_timer.start()
        else:
            c.restore_region(self._bg_info)
            self.info_ax.draw_artist(self.info)
            self._info_img = c.copy_from_bbox(self.info_ax.bbox)
            self._info_key, self._info_time = key, now
        if blit:
            c.blit(self.info_ax.bbox)

    def _info_later(self):
        self._info_timer = None
        if self._bg_info is not None:
            self._paint_info(blit=True)

    def snapshot(self, file, dpi=100):
        """Save the whole window, view and text included, to an image file."""
        arts = self.view.artists() + [self.info]
        self._quiet = True
        for a in arts:
            a.set_animated(False)
        self.fig.savefig(file, dpi=dpi)
        for a in arts:
            a.set_animated(True)
        self._quiet = False
        self.fig.canvas.draw_idle()


NO_WINDOW = """The viewer needs a window, and matplotlib cannot open one here ({why}).
Install a GUI toolkit for Python, then run the viewer again:
  Ubuntu, Debian:  sudo apt install python3-tk
  Fedora:          sudo dnf install python3-tkinter
  Arch:            sudo pacman -S tk
  any system:      pip install PyQt6
Python from python.org or Anaconda (Windows, macOS) already includes one.
On a remote machine there must also be a display (for example ssh -X).
Colab and Jupyter cannot show the viewer; there, use viewer.render() to make figures."""


def _pyplot_with_window():
    """pyplot, if matplotlib can open a window; otherwise say what to install, and exit."""
    try:
        import matplotlib.pyplot as plt
        backend = plt.get_backend()
        plt.switch_backend(backend)        # loads it: fails here if its toolkit is missing
    except ImportError as e:
        sys.exit(NO_WINDOW.format(why=e))
    if backend.lower() in ('agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template') or 'inline' in backend:
        sys.exit(NO_WINDOW.format(why=f'backend "{backend}" only writes image files'))
    return plt


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--impl', default='ex2', help='the module with your functions (default ex2)')
    p.add_argument('--mesh', help='an OBJ file to show instead of the airplane')
    p.add_argument('--scene', default='airplane', choices=['airplane', 'squares'])
    p.add_argument('--key0', default='0,0,0', help='Euler angles of key 0, e.g. 0,0,0')
    p.add_argument('--key1', default='0,0,0', help='Euler angles of key 1, e.g. 180,180,180')
    a = p.parse_args()
    use(a.impl)
    parse = lambda s: [float(v) for v in s.split(',')]
    plt = _pyplot_with_window()
    app = App(load_obj(a.mesh) if a.mesh else None, a.scene, parse(a.key0), parse(a.key1))
    plt.show()
    return app


if __name__ == '__main__':
    main()
