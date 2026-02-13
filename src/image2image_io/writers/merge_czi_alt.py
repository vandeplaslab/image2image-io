from __future__ import annotations
import xml.etree.ElementTree as ET
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


def stitch_czi_scenes_czifile(
    path: str,
    *,
    positions_px: list[tuple[int, int]] | None = None,
    scenes: list[int] | None = None,
    x_offset: tuple[int, ...] | list[int] | None = None,
    y_offset: tuple[int, ...] | list[int] | None = None,
    take_max_on_overlap: bool = True,
    z: int | None = 0,
    z_reduce: str = "max",
) -> tuple[np.ndarray, dict[int, ScenePlacement]]:
    """
    Stitches scenes with metadata and manual pixel offsets for fine-tuning.
    """
    with CziFile(path) as czi:
        data = czi.asarray()
        axes = getattr(czi, "axes", None).replace(" ", "")
        metadata_xml = czi.metadata()

    si = _get_axis_index(axes, "S")
    n_scenes = data.shape[si] if si is not None else 1
    scene_ids = [s for s in (scenes or range(n_scenes)) if 0 <= s < n_scenes]
    n_selected = len(scene_ids)

    # Validate manual offsets
    if x_offset and len(x_offset) != n_selected:
        raise ValueError(f"x_offset length ({len(x_offset)}) must match number of scenes ({n_selected})")
    if y_offset and len(y_offset) != n_selected:
        raise ValueError(f"y_offset length ({len(y_offset)}) must match number of scenes ({n_selected})")

    if positions_px is None:
        try:
            root = ET.fromstring(metadata_xml)
            scene_coords = {}
            for scene_node in root.findall(".//Scene"):
                idx_attr = scene_node.get("Index")
                if idx_attr is None:
                    continue
                idx = int(idx_attr)
                bbox = scene_node.find("BoundingBox")
                if bbox is not None:
                    scene_coords[idx] = (int(bbox.find("X").text), int(bbox.find("Y").text))

            positions_px = [scene_coords.get(s, (0, 0)) for s in scene_ids]
        except Exception as e:
            print(f"Metadata parsing failed: {e}")
            positions_px = [(i * 1000, 0) for i in range(n_selected)]

    # Apply manual offsets per scene
    final_positions = []
    for i in range(n_selected):
        base_x, base_y = positions_px[i]
        off_x = x_offset[i] if x_offset else 0
        off_y = y_offset[i] if y_offset else 0
        final_positions.append((base_x + off_x, base_y + off_y))

    scenes_cyx = []
    infos = {}
    min_x = min_y = 10**18
    max_x = max_y = -(10**18)

    for out_idx, s in enumerate(scene_ids):
        # Fresh copy of axes for each scene to avoid mismatch errors
        a, ax = data, axes
        if si is not None:
            a = np.take(a, indices=s, axis=si)
            ax = ax[:si] + ax[si + 1 :]

        a, ax = _reduce_z(a, ax, z=z, z_reduce=z_reduce)
        img = _to_cyx(a, ax)
        scenes_cyx.append(img)

        C, H, W = img.shape
        x0, y0 = final_positions[out_idx]
        infos[s] = ScenePlacement(scene=s, x0=x0, y0=y0, w=W, h=H)
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x0 + W), max(max_y, y0 + H)

    shift_x, shift_y = -min_x, -min_y
    stitched = np.zeros((scenes_cyx[0].shape[0], max_y - min_y, max_x - min_x), dtype=scenes_cyx[0].dtype)

    for out_idx, img in enumerate(scenes_cyx):
        s = scene_ids[out_idx]
        info = infos[s]
        xf, yf = info.x0 + shift_x, info.y0 + shift_y
        target = stitched[:, yf : yf + info.h, xf : xf + info.w]

        if take_max_on_overlap:
            np.maximum(target, img, out=target)
        else:
            target[...] = img

    return stitched, infos


# Helper dependencies from merge_czi.py
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
        return a, axes[:zi] + axes[zi + 1 :]
    # Reductions: max, mean, sum
    if z_reduce == "max":
        a = a.max(axis=zi)
    elif z_reduce == "mean":
        a = a.mean(axis=zi)
    elif z_reduce == "sum":
        a = a.sum(axis=zi)
    return a, axes[:zi] + axes[zi + 1 :]


def _to_cyx(scene: np.ndarray, axes: str) -> np.ndarray:
    """
    Convert an array with axes containing C,Y,X (and maybe others already removed)
    into (C,Y,X).
    """
    scene = np.squeeze(scene)

    if scene.ndim != len(axes):
        if scene.ndim == 2:
            return scene[None, ...]  # (1,Y,X)
        if scene.ndim == 3:
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
        perm = [i for i in range(scene.ndim) if i not in (yi, xi)] + [yi, xi]
        scene2 = np.transpose(scene, perm)
        scene2 = scene2[None, ...]
        if scene2.ndim != 3:
            scene2 = scene2.reshape((1,) + scene2.shape[-2:])
        return scene2

    perm = [ci, yi, xi] + [i for i in range(scene.ndim) if i not in (ci, yi, xi)]
    scene2 = np.transpose(scene, perm)
    scene2 = np.squeeze(scene2)

    if scene2.ndim == 2:
        scene2 = scene2[None, ...]
    if scene2.ndim != 3:
        raise ValueError(f"After reordering expected (C,Y,X), got {scene2.shape}")

    return scene2


def merge_czi(
    path: PathLike,
    output_dir: PathLike | None = None,
    scenes: list[int] | None = None,
    as_uint8: bool | None = None,
    tile_size: int = 1024,
    x_offset: tuple[int, ...] | list[int] | None = None,
    y_offset: tuple[int, ...] | list[int] | None = None,
) -> Path:
    """Merge CZI scenes into a single image, preserving channels."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array

    path = Path(path)
    output_dir = Path(output_dir if output_dir is not None else path.parent)

    czi = get_simple_reader(path, init_pyramid=False, auto_pyramid=False)
    is_rgb = czi.is_rgb
    channel_names = czi.channel_names
    resolution = czi.resolution
    czi.close()

    out_path = output_dir / path.with_name(path.stem + "_merged.ome.tiff").name
    if out_path.exists():
        return out_path

    stitched_array, _ = stitch_czi_scenes_czifile(
        str(path), scenes=scenes, take_max_on_overlap=not is_rgb, x_offset=x_offset, y_offset=y_offset
    )

    if stitched_array.ndim == 3 and np.argmin(stitched_array.shape) == 2 and is_rgb:
        stitched_array = np.moveaxis(stitched_array, 0, -1)

    write_ome_tiff_from_array(
        out_path,
        reader=None,
        array=stitched_array,
        resolution=resolution,
        channel_names=channel_names,
        tile_size=tile_size,
        as_uint8=as_uint8,
    )

    return out_path
