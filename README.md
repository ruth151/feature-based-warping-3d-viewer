# Feature-Based Image Warping & 3D Viewing

A from-scratch NumPy implementation of two classic computer-graphics topics:

1. **Image warping and morphing with feature lines** (Beier & Neely, SIGGRAPH 1992)
2. **The 3D pipeline**: transformations, rotations (Euler angles, quaternions, slerp), cameras and projections

Built for a university *Visual Computing* course. Everything is implemented with
NumPy only: no OpenCV, scikit-image or SciPy transform helpers.

> **Status:** work in progress. See the [roadmap](#roadmap).

## Features

### Part A: Warping and morphing
- Beier–Neely feature-line warp (`source_positions`, `warp_lines`), fully vectorized
- Bilinear and nearest-neighbor sampling
- Two-image morphing with cross-dissolve (`morph`)
- Least-squares affine fit and homography estimation (`fit_affine`, `fit_homography`)
- Backward warping by a 3x3 matrix (`warp_matrix`)

### Part B: 3D transformations and viewing
- Translation, scale, axis-angle (Rodrigues) and Euler rotations
- Quaternions: construction, multiplication, matrix conversion, slerp
- `look_at`, orthographic and perspective projection, viewport transform
- Reverse pipeline: casting a world-space ray through a pixel (`pixel_ray`)
- A chase camera that follows an airplane model (`chase_view`)

## Getting started

```bash
pip install numpy matplotlib
python warptool.py                        # interactive feature-line warp tool
python warptool.py morph_a.jpg morph_b.jpg  # morph between two images
python viewer.py                          # interactive 3D airplane viewer
python selftest.py                        # sanity-check the matrix conventions
```

The tools open a window, so run them on a desktop machine (not Colab/Jupyter).
On Linux you may need `sudo apt install python3-tk`.

## Project layout

| File | Purpose |
|---|---|
| `ex2.py` | My implementation of all the functions |
| `warptool.py` | Interactive tool for drawing feature lines and viewing the warp (course-provided) |
| `viewer.py`, `meshes.py` | Interactive 3D viewer and meshes (course-provided) |
| `selftest.py` | Convention checks (course-provided) |
| `bugs/` | Images used for the debugging exercise (course-provided) |

## Conventions

- Images are float arrays in `[0, 1]`; pixel `(row, col)` is centered at `(x, y) = (col, row)`
- Warping is backward: each output pixel samples the source at its computed position
- 3D transforms are 4x4 matrices on column vectors; the camera looks down `-z` (OpenGL style)
- Quaternions are `[w, x, y, z]`; angles are in degrees

## Roadmap

- [ ] Beier–Neely `source_positions` and `warp_lines`
- [ ] `morph`
- [ ] Affine / homography fitting and `warp_matrix`
- [ ] Part A experiments and write-up
- [ ] 3D transforms, quaternions, slerp
- [ ] Cameras, projection, `pixel_ray`, `chase_view`
- [ ] Part B experiments and write-up

## Credits

The interactive tools (`warptool.py`, `viewer.py`, `meshes.py`, `selftest.py`) and sample
images were provided by the course staff. The warp method follows Beier & Neely,
*Feature-Based Image Metamorphosis*, SIGGRAPH 1992.
