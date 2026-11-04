"""Exercise 2 warp tool: warping and morphing with feature lines, computed by your Part A functions.

Every warped image and every source position the tool shows comes from your ex2.py:
source_positions, warp_lines, morph, fit_affine, fit_homography and warp_matrix. The tool itself
only draws the images and the lines.

    python3 warptool.py                        the test image, test_image.png
    python3 warptool.py photo.png              warp another image
    python3 warptool.py face0.jpg face1.jpg    morph between two images of the same size
    python3 warptool.py --open warp-01.json    reopen the lines and settings saved by "save"

When morphing, "save frames" writes the folder my_images that A2c asks for: the two images
(morph0.png, morph1.png), their lines (lines.json) and the frames at t = 0, 0.25, 0.5, 0.75 and 1
(frame_000.png to frame_100.png), computed with your morph.

Lines: drag on an image to draw a pair of corresponding lines (they start out the same). Drag an
end of a line to move it, or its middle to move the whole line; the arrowhead marks Q. Click a
line to select it: its coordinates appear in the box under the images, where you can also type
them (Enter applies; with no line selected, Enter adds a pair). Delete removes the selected pair.
Probe (warping): point at a pixel X of the output to see its source position X'. Click to pin the
probe, move it with the arrow keys, and release it with Esc.
View: scroll to zoom, right-drag to pan, r to reset the view, h to hide the lines.
"""
import argparse
import importlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TEST_IMAGE = os.path.join(HERE, 'test_image.png')
FUNCTIONS = ['source_positions', 'warp_lines', 'morph', 'fit_affine', 'fit_homography', 'warp_matrix']
METHODS = ['Beier–Neely', 'affine fit', 'homography fit']
SAMPLING = ['bilinear', 'nearest']
SHOW = ['morph', 'cross-dissolve']
A_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
COLORS = ['#e8000b', '#1a53ff', '#00a33a', '#ff8c00', '#a51cd8', '#00a5b8', '#e6007e', '#8a6d00']
PROBE = '#ff00ff'
MAX_SIZE = 512

_impl = None


def use(module):
    """Choose the module that provides the functions: a module, or its name (default 'ex2')."""
    global _impl
    _impl = importlib.import_module(module) if isinstance(module, str) else module


def call(name, *args):
    """Call one of your functions, with a clear message if it is missing, fails or returns None."""
    if _impl is None:
        use('ex2')
    f = getattr(_impl, name, None)
    if f is None:
        raise NotImplementedError(f'{_impl.__name__}.{name} is missing')
    try:
        r = f(*args)
    except NotImplementedError:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f'{name} raised {type(e).__name__}: {e}') from e
    if r is None:
        raise NotImplementedError(f'{_impl.__name__}.{name} returned None (not implemented yet?)')
    return np.asarray(r, float)


def load_image(path, max_size=MAX_SIZE):
    """An image file as a float array in [0, 1], H x W (gray) or H x W x 3 (color), reduced to at
    most max_size pixels on its longer side. Returns the array and a note if it was reduced."""
    from PIL import Image
    im = Image.open(path)
    im = im.convert('L') if im.mode in ('1', 'L', 'LA', 'I', 'I;16', 'F') else im.convert('RGB')
    note = ''
    if max(im.size) > max_size:
        s = max_size / max(im.size)
        size = (max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s)))
        note = f'{os.path.basename(path)} reduced from {im.size[0]} × {im.size[1]} to {size[0]} × {size[1]}'
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, float) / 255, note


def fmt(v, d=2):
    """A number for the screen: no '-0.00', and integers without decimals."""
    v = float(v) + 0.0
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f'{v:.{d}f}'


def pt(P):
    """A point to two decimals, or without decimals when both coordinates round to integers."""
    s = [f'{v + 0.0:.2f}'.replace('-0.00', '0.00') for v in np.asarray(P, float)]
    if all(v.endswith('.00') for v in s):
        s = [v[:-3] for v in s]
    return '(' + ', '.join(s) + ')'


def num(v):
    """A parameter value: 1, 0.5, 0.05."""
    return f'{float(v):g}'


class App:
    def __init__(self, images, names, notes=(), state=None):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
        for k in list(matplotlib.rcParams):
            if k.startswith('keymap.') and k != 'keymap.quit':
                matplotlib.rcParams[k] = []
        matplotlib.rcParams['toolbar'] = 'None'

        self.morph = len(images) == 2
        self.imgs = images
        self.names = names
        self.H, self.W = images[0].shape[:2]
        self.lines = [np.zeros((0, 2, 2)), np.zeros((0, 2, 2))]
        self.a, self.b, self.p, self.t = 1.0, 2.0, 0.5, 0.5
        self.method, self.sampling, self.show = 0, 0, 0
        self.sel = None                    # index of the selected line pair
        self.probe = None                  # the probed output pixel (x, y)
        self.pinned = False
        self.show_lines = True
        self.drag = None
        self.M = None                      # the fitted matrix, for the affine and homography fits
        self.notes = list(notes)
        self.compute_time = 0.0
        self._syncing = False
        if state:
            self._restore(state)

        self.fig = plt.figure(figsize=(14, 8.4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('Exercise 2 warp tool')

        # the image panels: the images with their lines, then the result
        if self.morph:
            boxes = [[0.015 + 0.33 * i, 0.235, 0.31, 0.695] for i in range(3)]
            roles = [0, 1, None]
        else:
            boxes = [[0.03, 0.29, 0.45, 0.65], [0.52, 0.29, 0.45, 0.65]]
            roles = [0, 1]
        self.panels = []
        for box, role in zip(boxes, roles):
            ax = self.fig.add_axes(box)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#e8e8e8')
            img = self.imgs[role] if self.morph and role is not None else self.imgs[0]
            im = ax.imshow(self._display(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            msg = ax.text(0.5, 0.5, '', transform=ax.transAxes, ha='center', va='center', fontsize=11,
                          wrap=True, bbox=dict(boxstyle='round', fc='white', ec='#999999'), visible=False)
            self.panels.append(dict(ax=ax, im=im, role=role, msg=msg, artists=[]))
        self.out = self.panels[-1]         # the result panel, where the probe lives
        # the probe (warping only): + at the output pixel X, x at its source position X'
        label = dict(xytext=(7, 7), textcoords='offset points', color=PROBE, fontsize=12, fontweight='bold')
        self.src_probe = [self.panels[0]['ax'].plot([], [], 'x', color=PROBE, ms=12, mew=2.5)[0],
                          self.panels[0]['ax'].annotate('X′', (0, 0), **label)]
        self.out_probe = [self.out['ax'].plot([], [], '+', color=PROBE, ms=16, mew=2)[0],
                          self.out['ax'].annotate('X', (0, 0), **label)]
        self.reset_view()

        self.readout = self.fig.text(0.03 if not self.morph else 0.015, 0.25 if not self.morph else 0.215, '', fontsize=11,
                                     family='monospace', va='top')
        self.matrix = self.fig.text(0.52, 0.25, '', fontsize=11, family='monospace', va='top')

        # controls
        def slider(rect, label, lo, hi, v, step, fmt_):
            s = Slider(self.fig.add_axes(rect), label, lo, hi, valinit=v, valstep=step)
            s.valtext.set_text(fmt_(v))
            return s
        y0 = 0.13
        self.s_a = slider([0.06, y0, 0.22, 0.025], 'a', -2, 2, np.log10(self.a), np.log10(A_VALUES),
                          lambda v: num(10 ** v))
        self.s_b = slider([0.06, y0 - 0.04, 0.22, 0.025], 'b', 0, 4, self.b, 0.1, lambda v: f'{v:.1f}')
        self.s_p = slider([0.06, y0 - 0.08, 0.22, 0.025], 'p', 0, 1, self.p, 0.1, lambda v: f'{v:.1f}')
        self.s_a.on_changed(lambda v: self._on_param('a', 10 ** v, self.s_a, num(10 ** v)))
        self.s_b.on_changed(lambda v: self._on_param('b', v, self.s_b, f'{v:.1f}'))
        self.s_p.on_changed(lambda v: self._on_param('p', v, self.s_p, f'{v:.1f}'))
        if self.morph:
            self.s_t = slider([0.4, y0, 0.25, 0.025], 't', 0, 1, self.t, 0.01, lambda v: f'{v:.2f}')
            self.s_t.on_changed(lambda v: self._on_param('t', v, self.s_t, f'{v:.2f}'))
            self.r_show = RadioButtons(self.radio_axes([0.72, 0.07, 0.12, 0.075], 'show'),
                                       SHOW, active=self.show)
            self.r_show.on_clicked(lambda label: self._on_radio('show', SHOW.index(label)))
            tb_rect, label = [0.4, 0.045, 0.37, 0.04], 'line L₀ → L₁ '
        else:
            self.r_method = RadioButtons(self.radio_axes([0.36, 0.035, 0.13, 0.11], 'warp'),
                                         METHODS, active=self.method)
            self.r_method.on_clicked(lambda label: self._on_radio('method', METHODS.index(label)))
            self.r_sampling = RadioButtons(self.radio_axes([0.53, 0.07, 0.1, 0.075], 'sampling'),
                                           SAMPLING, active=self.sampling)
            self.r_sampling.on_clicked(lambda label: self._on_radio('sampling', SAMPLING.index(label)))
            tb_rect, label = [0.76, 0.115, 0.21, 0.04], 'line P′Q′ → PQ '
        self.tb = TextBox(self.fig.add_axes(tb_rect), label, initial='')
        self.tb.on_submit(self._on_text)
        save_rect = [0.88, 0.045, 0.09, 0.04] if not self.morph else [0.88, y0 - 0.01, 0.09, 0.04]
        self.b_save = Button(self.fig.add_axes(save_rect), 'save')
        self.b_save.on_clicked(lambda e: self.save())
        if self.morph:
            self.b_frames = Button(self.fig.add_axes([0.88, 0.055, 0.09, 0.04]), 'save frames')
            self.b_frames.on_clicked(lambda e: self.save_frames())
        hint = ['drag: new line pair', 'drag an end or the middle: move', 'click: select', 'Delete: remove',
                'scroll: zoom', 'r: reset view', 'h: hide lines']
        if not self.morph:
            hint[4:4] = ['point at the output: X′ (click pins it, arrows move it)']
        self.fig.text(0.5, 0.008, '   ·   '.join(hint), ha='center', fontsize=9, color='#595959')

        c = self.fig.canvas
        c.mpl_connect('key_press_event', self._on_key)
        c.mpl_connect('button_press_event', self._on_press)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('motion_notify_event', self._on_motion)
        c.mpl_connect('scroll_event', self._on_scroll)
        c.mpl_connect('axes_leave_event', self._on_leave)
        c.mpl_connect('figure_leave_event', self._on_leave)
        self._sync_text()
        self.compute()
        self.redraw()

    def radio_axes(self, rect, title):
        ax = self.fig.add_axes(rect, frameon=False)
        ax.set_title(title, loc='left', fontsize=10, pad=2)
        return ax

    # state ---------------------------------------------------------------------------------
    def _restore(self, s):
        get = lambda k: np.asarray(s[k], float).reshape(-1, 2, 2)
        if self.morph:
            self.lines = [get('lines0'), get('lines1')]
            self.t = float(s.get('t', self.t))
            self.show = SHOW.index(s.get('show', SHOW[0]))
        else:
            self.lines = [get('src_lines'), get('dst_lines')]
            self.method = METHODS.index(s.get('warp', METHODS[0]).replace('-', '–'))
            self.sampling = SAMPLING.index(s.get('sampling', SAMPLING[0]))
            if s.get('probe') is not None:
                self.probe, self.pinned = tuple(int(v) for v in s['probe']), True
        self.a, self.b, self.p = (float(s.get(k, d)) for k, d in (('a', self.a), ('b', self.b), ('p', self.p)))

    def state(self):
        lst = lambda L: [[[float(v) + 0.0 if v != int(v) else int(v) for v in P] for P in l] for l in L]
        s = dict(images=self.names)
        if self.morph:
            s.update(lines0=lst(self.lines[0]), lines1=lst(self.lines[1]), t=round(self.t, 4),
                     show=SHOW[self.show])
        else:
            s.update(src_lines=lst(self.lines[0]), dst_lines=lst(self.lines[1]), warp=METHODS[self.method],
                     sampling=SAMPLING[self.sampling])
        s.update(a=self.a, b=round(self.b, 4), p=round(self.p, 4))
        if not self.morph and self.pinned and self.probe is not None:
            s['probe'] = list(self.probe)
        return s

    def panel_lines(self, panel):
        """The lines drawn on a panel: none on the morph result."""
        return self.lines[panel['role']] if panel['role'] is not None else np.zeros((0, 2, 2))

    # computing (all by your functions) -------------------------------------------------------
    def _result(self):
        """The image in the result panel, and a note to show on it (or None)."""
        n = len(self.lines[0])
        L0, L1 = self.lines[0].copy(), self.lines[1].copy()
        self.M = None
        if self.morph:
            img0, img1 = self.imgs
            if self.show == 1 or n == 0:
                note = None if self.show == 1 else 'No lines yet: this is the cross-dissolve.\nDrag on img0 or img1 to draw a line pair.'
                return (1 - self.t) * img0 + self.t * img1, note
            return call('morph', img0, img1, L0, L1, self.t, self.a, self.b, self.p), None
        img = self.imgs[0]
        interp = SAMPLING[self.sampling]
        if self.method == 0:
            if n == 0:
                return img, 'No lines yet: the output is the source image.\nDrag on an image to draw a line pair.'
            return call('warp_lines', img, L0, L1, self.a, self.b, self.p, interp), None
        if n < 2:
            return None, f'The {METHODS[self.method]} needs at least two line pairs\n(their four endpoints are the correspondences).'
        f = 'fit_affine' if self.method == 1 else 'fit_homography'
        self.M = call(f, L0.reshape(-1, 2), L1.reshape(-1, 2))
        if self.M.shape != (3, 3):
            raise RuntimeError(f'{f} returned an array of shape {self.M.shape}, not 3 x 3')
        return call('warp_matrix', img, self.M, interp), None

    def compute(self):
        t0 = time.perf_counter()
        try:
            out, note = self._result()
            if out is not None and out.shape != self.imgs[0].shape:
                raise RuntimeError(f'the result has shape {out.shape}, but the image has shape {self.imgs[0].shape}')
        except (NotImplementedError, RuntimeError) as e:
            out, note = None, str(e)
        self.compute_time = time.perf_counter() - t0
        self.out['im'].set_data(self._display(out if out is not None else np.full(self.imgs[0].shape, 0.9)))
        self.out['msg'].set_text(note or '')
        self.out['msg'].set_visible(bool(note))
        self.update_probe()

    @staticmethod
    def _mark(marker, P):
        """Show the probe's marker and label at P, or hide them (P None)."""
        dot, label = marker
        dot.set_data([], []) if P is None else dot.set_data([P[0]], [P[1]])
        label.set_visible(P is not None)
        if P is not None:
            label.xy = tuple(P)

    def update_probe(self):
        """The source position X' of the probed output pixel X, from your functions."""
        self._mark(self.src_probe, None)
        self._mark(self.out_probe, self.probe)
        self.probe_text = ''
        if self.probe is None:
            return
        x, y = self.probe
        head = f'X = {pt(self.probe)}'
        try:
            if self.method == 0 and len(self.lines[0]):
                P = self._positions(np.array([[x, y]], float), self.lines[0], self.lines[1])
                self.probe_text = f'{head}    X′ = {pt(P)}'
            elif self.method != 0 and self.M is not None:
                h = np.linalg.solve(self.M, [x, y, 1.0])
                P = h[:2] / h[2]
                self.probe_text = f'{head}    X′ = M⁻¹X = {pt(P)}'
            else:
                self.probe_text = head
                return
            self._mark(self.src_probe, P)
        except (NotImplementedError, RuntimeError) as e:
            self.probe_text = f'{head}    {e}'

    def _positions(self, X, src, dst):
        P = call('source_positions', X, src.copy(), dst.copy(), self.a, self.b, self.p)
        if P.shape != (1, 2):
            raise RuntimeError(f'source_positions returned shape {P.shape} for 1 point, not (1, 2)')
        return P[0]

    # drawing -------------------------------------------------------------------------------
    def _display(self, img):
        return np.clip(img, 0, 1)

    def redraw(self):
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.patheffects as pe
        halo = [pe.Stroke(linewidth=4.5, foreground='white', alpha=0.7), pe.Normal()]
        halo_sel = [pe.Stroke(linewidth=7, foreground='white', alpha=0.9), pe.Normal()]
        for panel in self.panels:
            ax, arts = panel['ax'], panel['artists']
            n = len(self.panel_lines(panel))
            while len(arts) > n:
                for a in arts.pop():
                    a.remove()
            while len(arts) < n:
                arrow = FancyArrowPatch((0, 0), (1, 1), arrowstyle='-|>', mutation_scale=16, shrinkA=0,
                                        shrinkB=0, zorder=3)
                ax.add_patch(arrow)
                start = ax.plot([], [], 'o', zorder=4)[0]
                arts.append((arrow, start))
            for i, ((P, Q), (arrow, start)) in enumerate(zip(self.panel_lines(panel), arts)):
                c = COLORS[i % len(COLORS)]
                sel = i == self.sel
                arrow.set_positions(tuple(P), tuple(Q))
                arrow.set_color(c)
                arrow.set_linewidth(3 if sel else 1.8)
                arrow.set_path_effects(halo_sel if sel else halo)
                start.set_data([P[0]], [P[1]])
                start.set_color(c)
                start.set_markersize(8 if sel else 5.5)
                start.set_markeredgecolor('white')
                for a in (arrow, start):
                    a.set_visible(self.show_lines)
        if self.morph:
            self.panels[0]['ax'].set_title('img0: lines L₀', fontsize=12)
            self.panels[1]['ax'].set_title('img1: lines L₁', fontsize=12)
            what = 'cross-dissolve' if self.show == 1 else 'morph'
            self.out['ax'].set_title(f'{what} at t = {self.t:.2f}', fontsize=12)
        else:
            self.panels[0]['ax'].set_title('source: lines P′Q′', fontsize=12)
            what = METHODS[self.method]
            self.out['ax'].set_title(f'output ({what}): lines PQ', fontsize=12)
        self.readout.set_text('\n'.join(self.readout_lines(pinned=True) + self.notes))
        self.matrix.set_text(self.matrix_text())
        self._interpolation()
        self.fig.canvas.draw_idle()

    def readout_lines(self, pinned):
        """The probe, for the line under the images."""
        if not self.probe_text:
            return []
        return [self.probe_text + ('    (pinned)' if pinned and self.pinned else '')]

    def matrix_text(self):
        """The fitted matrix, under the output."""
        if self.M is None:
            return ''
        rows = ['[' + ' '.join(f'{np.round(v, 6) + 0.0:10.5g}' for v in r) + ' ]' for r in self.M]
        return f'M = {rows[0]}\n    {rows[1]}\n    {rows[2]}'

    def _interpolation(self):
        """Show pixels as squares when they are enlarged; smooth when the image is reduced."""
        ax = self.panels[0]['ax']
        x0, x1 = ax.get_xlim()
        scale = ax.bbox.width / abs(x1 - x0)
        for p in self.panels:
            p['im'].set_interpolation('nearest' if scale >= 1 else 'antialiased')

    def reset_view(self):
        """Fit the image and every line end."""
        pts = np.r_[[[-0.5, -0.5], [self.W - 0.5, self.H - 0.5]], self.lines[0].reshape(-1, 2),
                    self.lines[1].reshape(-1, 2)]
        lo, hi = pts.min(0), pts.max(0)
        m = 0.03 * (hi - lo).max()
        self._set_view(lo - m, hi + m)

    def _set_view(self, lo, hi):
        for p in self.panels:
            p['ax'].set_xlim(lo[0], hi[0])
            p['ax'].set_ylim(hi[1], lo[1])
            p['ax'].set_aspect('equal', adjustable='box')

    def _sync_text(self):
        self._syncing = True
        if self.sel is None:
            self.tb.set_val('')
        else:
            (P0, Q0), (P1, Q1) = self.lines[0][self.sel], self.lines[1][self.sel]
            v = lambda P: f'{fmt(P[0])} {fmt(P[1])}'
            self.tb.set_val(f'{v(P0)} {v(Q0)}  →  {v(P1)} {v(Q1)}')
        if not self.tb.capturekeystrokes:
            self.tb.cursor.set_visible(False)  # set_val shows the cursor even when not typing
        self._syncing = False

    # editing -------------------------------------------------------------------------------
    def _panel(self, ev):
        for p in self.panels:
            if ev.inaxes is p['ax']:
                return p
        return None

    def _data(self, panel, ev):
        return panel['ax'].transData.inverted().transform((ev.x, ev.y))

    def _px(self, points):
        """Screen points to the pixels of mouse events (twice as many on a Retina screen)."""
        return points * self.fig.dpi / 72

    def _hit(self, panel, ev):
        """(index, 'P' | 'Q' | 'line') of the line under the mouse in this panel, or None. An end
        within 12 points wins (so the arrowhead grabs Q); elsewhere along the line, the whole line."""
        L = self.panel_lines(panel)
        if not len(L) or not self.show_lines:
            return None
        S = panel['ax'].transData.transform(L.reshape(-1, 2)).reshape(-1, 2, 2)
        m = np.array([ev.x, ev.y])
        ends = np.linalg.norm(S - m, axis=2)            # N x 2: the distances to P and Q
        i, k = np.unravel_index(np.argmin(ends), ends.shape)
        if ends[i, k] < self._px(12):
            return int(i), 'PQ'[k]
        d = S[:, 1] - S[:, 0]
        u = np.clip(np.einsum('ij,ij->i', m - S[:, 0], d) / np.maximum(np.einsum('ij,ij->i', d, d), 1e-9), 0, 1)
        dist = np.linalg.norm(S[:, 0] + u[:, None] * d - m, axis=1)
        i = int(np.argmin(dist))
        return (i, 'line') if dist[i] < self._px(6) else None

    def select(self, i):
        self.sel = i
        self._sync_text()

    def _on_press(self, ev):
        panel = self._panel(ev)
        if panel is None or self.tb.capturekeystrokes:
            return
        self.notes = []                    # notes last until the next action
        if ev.button == 3 or (ev.button == 1 and ev.key == 'shift'):
            ax = panel['ax']
            self.drag = dict(what='pan', start=(ev.x, ev.y), xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                             inv=ax.transData.inverted().frozen())
            return
        if ev.button != 1:
            return
        hit = self._hit(panel, ev)
        start = np.round(self._data(panel, ev))
        self.drag = dict(panel=panel, start_px=(ev.x, ev.y), start=start, moved=False, hit=hit)
        if hit and panel['role'] is not None:
            i, what = hit
            self.drag.update(what=what, i=i, orig=self.lines[panel['role']][i].copy())
            self.select(i)
            self.redraw()
        else:
            self.drag['what'] = 'new' if panel['role'] is not None else 'click'

    def _on_motion(self, ev):
        d = self.drag
        if d is None:
            if not self.morph and not self.pinned and not self.tb.capturekeystrokes:
                if self._panel(ev) is self.out:
                    self.set_probe(self._data(self.out, ev))
                elif self.probe is not None:  # the probe follows the mouse only over the output
                    self.clear_probe()
            return
        if d['what'] == 'pan':
            if ev.x is None:
                return
            p0, p1 = d['inv'].transform([d['start'], (ev.x, ev.y)])
            dx, dy = p0 - p1
            self._set_view(np.array([d['xlim'][0] + dx, d['ylim'][1] + dy]),
                           np.array([d['xlim'][1] + dx, d['ylim'][0] + dy]))
            self._interpolation()
            self.fig.canvas.draw_idle()
            return
        if not d['moved']:
            if np.hypot(ev.x - d['start_px'][0], ev.y - d['start_px'][1]) < self._px(3):
                return
            d['moved'] = True
        if d['what'] == 'click':
            return
        panel, role = d['panel'], d['panel']['role']
        q = np.round(self._data(panel, ev))
        L = self.lines[role]
        if d['what'] == 'new':
            if np.array_equal(q, d['start']):
                return
            line = np.array([[d['start'], q]])
            self.lines = [np.r_[self.lines[0], line], np.r_[self.lines[1], line]]
            d.update(what='Q', i=len(self.lines[0]) - 1, both=True)
            self.select(d['i'])
        elif d['what'] in ('P', 'Q'):
            k = 0 if d['what'] == 'P' else 1
            if not d.get('both'):              # an existing end moves with the mouse, without jumping to it
                q = d['orig'][k] + (q - d['start'])
            if np.array_equal(q, L[d['i'], 1 - k]):
                return
            for r in ((0, 1) if d.get('both') else (role,)):   # a new line: both copies follow
                self.lines[r][d['i'], k] = q
        else:
            L[d['i']] = d['orig'] + (q - d['start'])
        self._sync_text()
        if self.compute_time < 0.08:       # fast enough to follow the mouse
            self.compute()
            d['computed'] = True
        else:
            d['computed'] = False
        self.redraw()

    def _on_release(self, ev):
        d, self.drag = self.drag, None
        if d is None or d['what'] == 'pan':
            return
        if not d['moved']:
            if d['hit']:
                self.select(d['hit'][0])
            elif d['panel'] is self.out and not self.morph:
                self.select(None)
                self.pinned = True
                self.set_probe(d['start'], redraw=False)
            else:
                self.select(None)
            self.redraw()
            return
        if d['what'] != 'click' and not d.get('computed'):
            self.compute()
        self.redraw()

    def clear_probe(self):
        self.probe, self.pinned = None, False
        self.update_probe()
        self.redraw()

    def _on_leave(self, ev):
        if self.drag is None and not self.pinned and self.probe is not None:
            self.clear_probe()

    def set_probe(self, xy, redraw=True):
        x = int(np.clip(np.round(xy[0]), 0, self.W - 1))
        y = int(np.clip(np.round(xy[1]), 0, self.H - 1))
        if self.probe != (x, y):
            self.probe = (x, y)
            self.update_probe()
            if redraw:
                self.redraw()

    def _on_scroll(self, ev):
        panel = self._panel(ev)
        if panel is None:
            return
        f = 1 / 1.2 if ev.button == 'up' else 1.2
        c = self._data(panel, ev)
        ax = panel['ax']
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        lo = c + (np.array([x0, y0]) - c) * f
        hi = c + (np.array([x1, y1]) - c) * f
        self._set_view(lo, hi)
        self._interpolation()
        self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        if self.tb.capturekeystrokes:
            return
        self.notes = []
        k = ev.key
        if k in ('delete', 'backspace') and self.sel is not None:
            keep = np.arange(len(self.lines[0])) != self.sel
            self.lines = [self.lines[0][keep], self.lines[1][keep]]
            self.select(None)
            self.compute()
        elif k in ('left', 'right', 'up', 'down') and not self.morph:
            x, y = self.probe if self.probe is not None else (self.W // 2, self.H // 2)
            dx, dy = dict(left=(-1, 0), right=(1, 0), up=(0, -1), down=(0, 1))[k]
            self.pinned = True
            self.set_probe((x + dx, y + dy), redraw=False)
        elif k == 'escape':
            self.pinned = False
            self.probe = None
            self.select(None)
            self.update_probe()
        elif k == 'h':
            self.show_lines = not self.show_lines
        elif k == 'r':
            self.reset_view()
        else:
            return
        self.redraw()

    def _on_text(self, text):
        if self._syncing:
            return
        nums = [float(v) for v in re.findall(r'[-+]?\d+(?:\.\d*)?|[-+]?\.\d+', text)]
        if not nums:
            return
        if len(nums) != 8:
            what = 'both ends in img0, then in img1' if self.morph else 'P′ and Q′, then of P and Q'
            self.notes = [f'Type 8 numbers: x y of {what} (got {len(nums)}).']
            self.redraw()
            return
        pair = np.array(nums).reshape(2, 2, 2)
        if np.any(np.all(pair[:, 0] == pair[:, 1], axis=1)):
            self.notes = ['The two ends of a line must differ.']
            self.redraw()
            return
        self.notes = []
        if self.sel is None:
            self.lines = [np.r_[self.lines[0], pair[None, 0]], np.r_[self.lines[1], pair[None, 1]]]
            self.sel = len(self.lines[0]) - 1
        else:
            self.lines[0][self.sel], self.lines[1][self.sel] = pair
        self._sync_text()
        lo, hi = np.array([self.panels[0]['ax'].get_xlim()[0], self.panels[0]['ax'].get_ylim()[1]]), \
            np.array([self.panels[0]['ax'].get_xlim()[1], self.panels[0]['ax'].get_ylim()[0]])
        ends = pair.reshape(-1, 2)
        if np.any(ends < lo) or np.any(ends > hi):
            self.reset_view()
        self.compute()
        self.redraw()

    def _on_param(self, name, v, slider, text):
        slider.valtext.set_text(text)
        if self._syncing:
            return
        setattr(self, name, float(v))
        self.compute()
        self.redraw()

    def _on_radio(self, name, i):
        setattr(self, name, i)
        self.compute()
        self.redraw()

    # saving --------------------------------------------------------------------------------
    def caption(self):
        if self.morph:
            what = 'cross-dissolve' if self.show == 1 else f'morph, a = {num(self.a)}, b = {num(self.b)}, p = {num(self.p)}'
            return f'{what}, t = {self.t:.2f}'
        if self.method == 0:
            return f'Beier–Neely, a = {num(self.a)}, b = {num(self.b)}, p = {num(self.p)}, {SAMPLING[self.sampling]} sampling'
        return f'{METHODS[self.method]}, {SAMPLING[self.sampling]} sampling'

    def save(self):
        """Save the image panels (with the lines and a caption) as a PNG, and the lines and
        settings as JSON, under the next free name in the current folder."""
        from matplotlib.transforms import Bbox
        base = 'morph' if self.morph else 'warp'
        n = 1
        while os.path.exists(f'{base}-{n:02d}.png') or os.path.exists(f'{base}-{n:02d}.json'):
            n += 1
        png, js = f'{base}-{n:02d}.png', f'{base}-{n:02d}.json'
        with open(js, 'w') as f:
            f.write(to_json(self.state()))
        hover = self.probe is not None and not self.pinned
        if hover:
            self.probe = None
            self.update_probe()
        notes, self.notes = self.notes, []
        self.redraw()
        self.readout.set_text('\n'.join([self.caption()] + self.readout_lines(pinned=False)))
        controls = [ax for ax in self.fig.axes if all(ax is not p['ax'] for p in self.panels)]
        controls += [t for t in self.fig.texts if t is not self.readout and t is not self.matrix]
        for c in controls:
            c.set_visible(False)
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        boxes = [p['ax'].get_tightbbox(r) for p in self.panels] + [t.get_window_extent(r) for t in
                                                                     (self.readout, self.matrix) if t.get_text()]
        bb = Bbox.union(boxes).transformed(self.fig.dpi_scale_trans.inverted()).padded(0.1)
        self.fig.savefig(png, dpi=150, bbox_inches=bb)
        for c in controls:
            c.set_visible(True)
        self.notes = notes + [f'saved {png} and {js}']
        self.redraw()


    FRAMES = (0, 0.25, 0.5, 0.75, 1)

    def save_frames(self, folder='my_images'):
        """Write the A2c folder: morph0.png and morph1.png (the images as the tool uses them),
        lines.json (the lines and settings), and frame_000.png ... frame_100.png, computed with
        your morph at t = 0, 0.25, 0.5, 0.75 and 1."""
        from PIL import Image
        if not len(self.lines[0]):
            self.notes = ['Draw the lines first: the frames are computed with them.']
            self.redraw()
            return
        frames = []
        try:
            for t in self.FRAMES:
                f = call('morph', self.imgs[0], self.imgs[1], self.lines[0].copy(), self.lines[1].copy(),
                         t, self.a, self.b, self.p)
                if f.shape != self.imgs[0].shape:
                    raise RuntimeError(f'morph returned shape {f.shape} at t = {t}, not {self.imgs[0].shape}')
                frames.append((f'frame_{round(100 * t):03d}.png', f))
        except (NotImplementedError, RuntimeError) as e:
            self.notes = [f'no frames saved: {e}']
            self.redraw()
            return
        os.makedirs(folder, exist_ok=True)
        to8 = lambda a: Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8))
        for name, img in [('morph0.png', self.imgs[0]), ('morph1.png', self.imgs[1])] + frames:
            to8(img).save(os.path.join(folder, name))
        state = self.state()
        state['images'] = ['morph0.png', 'morph1.png']
        with open(os.path.join(folder, 'lines.json'), 'w') as f:
            f.write(to_json(state))
        self.notes = [f'saved {folder}/: morph0.png, morph1.png, lines.json and {len(frames)} frames '
                      f'(a = {num(self.a)}, b = {num(self.b)}, p = {num(self.p)})']
        self.redraw()


def to_json(state):
    """The state as JSON, one line pair per row."""
    rows = []
    for k, v in state.items():
        if k.endswith('lines') or k in ('lines0', 'lines1'):
            inner = ',\n'.join('    ' + json.dumps(line) for line in v)
            rows.append(f'  "{k}": [\n{inner}\n  ]' if v else f'  "{k}": []')
        else:
            rows.append(f'  "{k}": {json.dumps(v, ensure_ascii=False)}')
    return '{\n' + ',\n'.join(rows) + '\n}\n'


def open_images(paths, max_size):
    imgs, notes = [], []
    for path in paths:
        img, note = load_image(path, max_size)
        imgs.append(img)
        if note:
            notes.append(note)
    if len(imgs) == 2:
        if imgs[0].shape[:2] != imgs[1].shape[:2]:
            sys.exit(f'The two images must have the same size: {paths[0]} is {imgs[0].shape[1]} × {imgs[0].shape[0]}, '
                     f'{paths[1]} is {imgs[1].shape[1]} × {imgs[1].shape[0]} (after reducing to at most {max_size}).')
        if imgs[0].ndim != imgs[1].ndim:      # one gray, one color: make both color
            imgs = [np.dstack([im] * 3) if im.ndim == 2 else im for im in imgs]
    return imgs, notes


NO_WINDOW = """The warp tool needs a window, and matplotlib cannot open one here ({why}).
Install a GUI toolkit for Python, then run the tool again:
  Ubuntu, Debian:  sudo apt install python3-tk
  Fedora:          sudo dnf install python3-tkinter
  Arch:            sudo pacman -S tk
  any system:      pip install PyQt6
Python from python.org or Anaconda (Windows, macOS) already includes one.
On a remote machine there must also be a display (for example ssh -X).
Colab and Jupyter cannot show the tool; there, call your functions and plot the results."""


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
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images', nargs='*', help='one image to warp, or two images of the same size to morph')
    ap.add_argument('--open', help='a JSON file saved by the tool: its lines and settings (and images)')
    ap.add_argument('--impl', default='ex2', help='the module with your functions (default ex2)')
    ap.add_argument('--max-size', type=int, default=MAX_SIZE,
                    help=f'reduce larger images to this many pixels on the longer side (default {MAX_SIZE})')
    a = ap.parse_args()
    if len(a.images) > 2:
        ap.error('give one image to warp, or two to morph')
    try:
        use(a.impl)
    except ModuleNotFoundError as e:
        if e.name != a.impl:
            raise
        sys.exit(f'No {a.impl}.py here: put your ex2.py in {HERE}, next to warptool.py.')
    state = None
    paths = a.images
    if a.open:
        with open(a.open) as f:
            state = json.load(f)
        if not paths:
            base = os.path.dirname(os.path.abspath(a.open))
            paths = [p if os.path.exists(p) else os.path.join(base, p) if os.path.exists(os.path.join(base, p))
                     else os.path.join(HERE, p) for p in state.get('images', [])]
        if len(paths) != (2 if 'lines0' in state else 1):
            sys.exit(f'{a.open} holds {"a morph" if "lines0" in state else "a warp"}: give '
                     f'{"two images" if "lines0" in state else "one image"}, or none to use the ones it names.')
    if not paths:
        paths = [TEST_IMAGE]
    imgs, notes = open_images(paths, a.max_size)
    names = [os.path.basename(p) if os.path.abspath(p) == TEST_IMAGE else p for p in paths]
    plt = _pyplot_with_window()
    app = App(imgs, names, notes, state)
    plt.show()
    return app


if __name__ == '__main__':
    main()
