"""Convert and stitch CZI scenes using czifile, preserving channel data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from czifile import CziFile
from koyo.path import PathLike


@dataclass
class ScenePlacement:
    """Placement info for a scene in the stitched image."""

    scene: int
    x0: int
    y0: int
    w: int
    h: int


def _get_axis_index(axes: str, key: str) -> int | None:
    try:
        return axes.index(key)
    except ValueError:
        return None


def _reduce_z(a: np.ndarray, axes: str, z: int | None, z_reduce: str) -> tuple[np.ndarray, str]:
    zi = _get_axis_index(axes, "Z")
    if zi is None:
        return a, axes

    if z is not None:
        a = np.take(a, indices=z, axis=zi)
        axes = axes[:zi] + axes[zi + 1 :]
        return a, axes

    if z_reduce == "max":
        a = a.max(axis=zi)
    elif z_reduce == "mean":
        a = a.mean(axis=zi)
    elif z_reduce == "sum":
        a = a.sum(axis=zi)
    else:
        raise ValueError(f"Unknown z_reduce={z_reduce!r}")
    axes = axes[:zi] + axes[zi + 1 :]
    return a, axes


def _to_cyx(scene: np.ndarray, axes: str) -> np.ndarray:
    """
    Convert an array with axes containing C,Y,X (and maybe others already removed)
    into (C,Y,X).
    """
    # S and Z should already be removed at this point
    # We'll squeeze leftover singleton dims.
    scene = np.squeeze(scene)

    # Recompute axes after squeeze is hard; instead, just move the known axes
    # based on current 'axes' string length. We'll do a safer approach:
    # build a list of axes present and then move them.
    # If squeeze removed dims, assume they were singleton non-CYX dims.
    # So we first try to locate C/Y/X positions in the *unsqueezed* scene.

    # Best approach: move axes before squeezing.
    # We'll implement it here: if scene ndim matches len(axes), use that.
    if scene.ndim != len(axes):
        # fallback: handle common simple cases
        if scene.ndim == 2:
            return scene[None, ...]  # (1,Y,X)
        if scene.ndim == 3:
            # assume already (C,Y,X) or (Y,X,C)
            if scene.shape[0] <= 32:
                return scene
            return np.moveaxis(scene, -1, 0)
        raise ValueError(f"Cannot align axes={axes!r} with scene.ndim={scene.ndim}, shape={scene.shape}")

    ci = _get_axis_index(axes, "C")
    yi = _get_axis_index(axes, "Y")
    xi = _get_axis_index(axes, "X")
    if yi is None or xi is None:
        raise ValueError(f"Need Y and X axes, got axes={axes!r}")

    if ci is None:
        # no C axis -> single channel
        # move Y,X to the end then add C=1
        perm = [i for i in range(scene.ndim) if i not in (yi, xi)] + [yi, xi]
        scene2 = np.transpose(scene, perm)
        scene2 = scene2[None, ...]  # (1, Y, X) plus any leftover dims (shouldn't happen)
        if scene2.ndim != 3:
            # collapse any leftover dims
            scene2 = scene2.reshape((1,) + scene2.shape[-2:])
        return scene2

    # Permute to (C, Y, X) dropping any other dims by flattening them (should be singleton)
    perm = [ci, yi, xi] + [i for i in range(scene.ndim) if i not in (ci, yi, xi)]
    scene2 = np.transpose(scene, perm)

    # If there are extra dims left, they should be 1; squeeze them away
    scene2 = np.squeeze(scene2)

    if scene2.ndim == 2:
        scene2 = scene2[None, ...]
    if scene2.ndim != 3:
        raise ValueError(f"After reordering expected (C,Y,X), got {scene2.shape}")

    return scene2


def stitch_czi_scenes_czifile(
    path: str,
    *,
    positions_px: list[tuple[int, int]] | None = None,
    scenes: list[int] | None = None,
    take_max_on_overlap: bool = True,
    z: int | None = 0,
    z_reduce: str = "max",
) -> tuple[np.ndarray, dict[int, ScenePlacement]]:
    """
    Stitches scenes using the CZI array's explicit axes.
    NOTE: 'positions_px' is optional; if None, scenes are placed in a row (debug default).
          If you already have positions in the same coordinate space, pass them as pixel offsets.
          'scenes' is a list of scene indices to export; None means all scenes.
    """
    with CziFile(path) as czi:
        data = czi.asarray()
        axes = getattr(czi, "axes", None)
        if axes is None:
            raise ValueError(
                "Your czifile version doesn't expose czi.axes; upgrade czifile or provide axis order manually."
            )

    axes = axes.replace(" ", "")
    si = _get_axis_index(axes, "S")
    if si is None:
        # no scene axis; treat the whole image as one scene
        si = None
        n_scenes = 1
    else:
        n_scenes = data.shape[si]

    # Which scenes to export
    if scenes is None:
        scene_ids = list(range(n_scenes))
    else:
        # de-duplicate while preserving order
        seen: set[int] = set()
        scene_ids = []
        for s in scenes:
            if s in seen:
                continue
            if s < 0 or s >= n_scenes:
                raise ValueError(f"Scene index out of range: {s} (valid 0..{n_scenes - 1})")
            seen.add(s)
            scene_ids.append(s)

    n_selected = len(scene_ids)
    if n_selected == 0:
        raise ValueError("No scenes selected.")

    # Debug fallback placement if you don't provide positions yet
    if positions_px is None:
        positions_px = [(0, 0)]
        if n_selected > 1:
            # lay them out side-by-side just so you can verify channels are correct
            positions_px = [(i * 1000, 0) for i in range(n_selected)]  # placeholder spacing

    if len(positions_px) != n_selected:
        raise ValueError(f"positions_px must have length {n_selected}, got {len(positions_px)}")

    scenes_cyx: list[np.ndarray] = []
    infos: dict[int, ScenePlacement] = {}

    # First pass: compute bounds
    min_x = 10**18
    min_y = 10**18
    max_x = -(10**18)
    max_y = -(10**18)

    for out_idx, s in enumerate(scene_ids):
        a = data
        ax = axes

        # pick scene
        if si is not None:
            a = np.take(a, indices=s, axis=si)
            ax = ax[:si] + ax[si + 1 :]

        # handle Z if present
        a, ax = _reduce_z(a, ax, z=z, z_reduce=z_reduce)

        img = _to_cyx(a, ax)
        scenes_cyx.append(img)

        C, H, W = img.shape
        x0, y0 = positions_px[out_idx]

        infos[s] = ScenePlacement(scene=s, x0=x0, y0=y0, w=W, h=H)
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x0 + W)
        max_y = max(max_y, y0 + H)

    # shift to positive
    shift_x = -min_x
    shift_y = -min_y
    out_w = max_x - min_x
    out_h = max_y - min_y

    C = scenes_cyx[0].shape[0]
    stitched = np.zeros((C, out_h, out_w), dtype=scenes_cyx[0].dtype)

    # paste
    for out_idx, img in enumerate(scenes_cyx):
        s = scene_ids[out_idx]
        info = infos[s]
        x0 = info.x0 + shift_x
        y0 = info.y0 + shift_y
        target = stitched[:, y0 : y0 + info.h, x0 : x0 + info.w]
        infos[s] = ScenePlacement(scene=s, x0=x0, y0=y0, w=info.w, h=info.h)

        if take_max_on_overlap:
            np.maximum(target, img, out=target)
        else:
            target[...] = img
    return stitched, infos


def merge_czi(path: PathLike) -> Path:
    """Merge CZI scenes into a single image, preserving channels."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array

    path = Path(path)
    czi = get_simple_reader(path, init_pyramid=False, auto_pyramid=False)
    is_rgb = czi.is_rgb
    channel_names = czi.channel_names
    resolution = czi.resolution
    czi.close()

    stitched_array, _ = stitch_czi_scenes_czifile(str(path), take_max_on_overlap=not is_rgb)
    # this ensures that aces are correctly ordered for writing
    if stitched_array.ndim == 3 and np.argmin(stitched_array.shape) == 2 and is_rgb:
        stitched_array = np.moveaxis(stitched_array, 0, -1)  # (Y,X,C) from (C,Y,X)

    out_path = path.with_name(path.stem + "_merged.ome.tiff")
    write_ome_tiff_from_array(
        out_path,
        reader=None,
        array=stitched_array,
        resolution=resolution,
        channel_names=channel_names,
    )

    return out_path
