"""Geo-JSON reader."""

from __future__ import annotations

import json
import typing as ty
import zipfile
from pathlib import Path

import cv2
import geojson
import numpy as np
from koyo.typing import PathLike
from shapely.geometry import Polygon

GJ_SHAPE_TYPE = {
    "polygon": geojson.Polygon,
    "multipolygon": geojson.MultiPolygon,
    "point": geojson.Point,
    "multipoint": geojson.MultiPoint,
    "multilinestring": geojson.MultiLineString,
    "linestring": geojson.LineString,
}
GJ_SHAPE_TYPE_NAME = {
    "polygon": "Polygon",
    "multipolygon": "MultiPolygon",
    "point": "Point",
    "multipoint": "MultiPoint",
    "multilinestring": "MultiLineString",
    "linestring": "LineString",
}


def get_int_dtype(value: int) -> np.dtype:
    """Determine appropriate bit precision for indexed image.

    Parameters
    ----------
    value:int
        number of shapes

    Returns
    -------
    dtype:np.dtype
        Appropriate data type for the number of masks.
    """
    if value <= np.iinfo(np.uint8).max:
        return np.uint8
    if value <= np.iinfo(np.uint16).max:
        return np.uint16
    if value <= np.iinfo(np.uint32).max:
        return np.int32
    else:
        raise ValueError("Too many shapes")


def geojson_to_numpy(gj: dict) -> tuple[list[dict], bool]:
    """
    Convert geojson representation to np.ndarray representation of shape.

    Parameters
    ----------
    gj : dict
        GeoJSON data stored as python dict

    Returns
    -------
    dict
        containing keys
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    is_points = False
    if gj["geometry"].get("type") == "MultiPolygon":
        pts = []
        for geo in gj["geometry"].get("coordinates"):
            for geo_ in geo:
                geo_ = np.squeeze(geo_)
                pts.append(np.asarray(np.asarray(Polygon(geo_).exterior.coords)))
    elif gj["geometry"].get("type") == "Polygon":
        try:
            coordinates = np.squeeze(np.asarray(gj["geometry"].get("coordinates")))
            pts = np.asarray(Polygon(coordinates).exterior.coords)
        except (ValueError, TypeError):
            pts = []
            for poly in gj["geometry"].get("coordinates"):
                pts.append(np.asarray(Polygon(np.asarray(np.squeeze(poly))).exterior.coords))
        # pts = np.squeeze(np.asarray(gj["geometry"].get("coordinates")))
    elif gj["geometry"].get("type") == "Point":
        pts = np.expand_dims(np.asarray(gj["geometry"].get("coordinates")), 0)
        is_points = True
    elif gj["geometry"].get("type") == "MultiPoint":
        pts = np.asarray(gj["geometry"].get("coordinates"))
        is_points = True
    elif gj["geometry"].get("type") == "LineString":
        pts = np.asarray(gj["geometry"].get("coordinates"))
    else:
        raise ValueError(f"GeoJSON type {gj['geometry'].get('type')} not supported")

    if "properties" not in gj:
        shape_name = "unnamed"
    else:
        if not gj["properties"].get("classification"):
            shape_name = "unnamed"
        else:
            shape_name = gj["properties"].get("classification").get("name")

    if isinstance(pts, list):
        return [
            {
                "array": np.asarray(pt).astype(np.float32),
                "shape_type": gj["geometry"].get("type"),
                "shape_name": shape_name,
            }
            for pt in pts
        ], is_points
    # elif len(pts.shape) == 1:
    #     return [
    #         {
    #             "array": np.asarray(pts[0]).astype(np.float32),
    #             "shape_type": gj["geometry"].get("type"),
    #             "shape_name": shape_name,
    #         }
    #     ]
    else:
        return [
            {
                "array": pts.astype(np.float32),
                "shape_type": gj["geometry"].get("type"),
                "shape_name": shape_name,
            }
        ], is_points


def add_unnamed(gj: dict) -> dict:
    """Add unnamed object."""
    if "properties" not in gj:
        gj["properties"] = {"classification": {"name": "unnamed"}}
    else:
        if not gj["properties"].get("classification"):
            gj["properties"].update({"classification": {"name": "unnamed"}})
    return gj


def read_geojson(json_file: PathLike) -> tuple[list, list, bool]:
    """Read GeoJSON files (and some QuPath metadata).

    Parameters
    ----------
    json_file : str
        file path of QuPath exported GeoJSON

    Returns
    -------
    gj_data : list
        GeoJSON information
    shapes_np : list
        GeoJSON information stored in np.ndarray
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    if Path(json_file).suffix != ".zip":
        gj_data = json.load(open(json_file))
    else:
        with zipfile.ZipFile(json_file, "r") as z:
            for filename in z.namelist():
                with z.open(filename) as f:
                    data = f.read()
                    gj_data = json.loads(data.decode("utf-8"))
    return _parse_geojson_data(gj_data)


def _parse_geojson_data(gj_data: dict | list) -> tuple[list, list, bool]:
    if isinstance(gj_data, dict):
        # handle GeoPandas GeoJSON
        if "type" in gj_data and "features" in gj_data:
            gj_data = gj_data["features"]
        else:
            gj_data = [gj_data]

    shapes_np, is_points = [], False
    for gj in gj_data:
        data, is_points = geojson_to_numpy(gj)
        shapes_np.extend(data)
    gj_data = [add_unnamed(gj) for gj in gj_data]
    return gj_data, shapes_np, is_points


def numpy_to_geojson(
    np_array: np.ndarray, shape_type: str = "polygon", shape_name: str = "unnamed"
) -> tuple[dict, dict]:
    """Convert np.ndarray to GeoJSON dict.

    Parameters
    ----------
    np_array: np.ndarray
        coordinates of data
    shape_type:str
        GeoJSON shape type
    shape_name:str
        Name of the shape

    Returns
    -------
    shape_gj : dict
        GeoJSON information
    shape_np : dict
        GeoJSON information stored in np.ndarray
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    sh_type = shape_type.lower()

    gj_func = GJ_SHAPE_TYPE[sh_type]
    if sh_type == "polygon":
        np_array = np.vstack([np_array, np_array[0, :]])
        geometry = gj_func([np_array.tolist()])
    elif sh_type in ["multipoint", "linestring"]:
        geometry = gj_func(np_array.transpose().tolist())
    else:
        geometry = gj_func(np_array.tolist())

    shape_gj = {
        "type": "Feature",
        "id": "annotation",
        "geometry": geometry,
        "properties": {
            "classification": {"name": shape_name, "colorRGB": -1},
            "isLocked": False,
        },
    }

    shape_np = {
        "array": np_array,
        "shape_type": shape_type.lower(),
        "shape_name": shape_name,
    }
    return shape_gj, shape_np


def shape_reader(shape_data: list, **kwargs: ty.Any) -> tuple[dict, dict, bool]:
    """Read shape data for transformation.

    Shape data is stored as numpy arrays for operations but also as GeoJSON
    to contain metadata and interface with QuPath.

    Parameters
    ----------
    shape_data: list of np.ndarray or str
        if str, will read data as GeoJSON file, if np.ndarray with assume
        it is coordinates
    kwargs
        keyword args passed to np_to_geojson convert

    Returns
    -------
    shapes_gj: list of dicts
        list of dicts of GeoJSON information
    shapes_np: list of dicts
        list of dicts of GeoJSON information stored in np.ndarray
        "array": np.ndarray - x,y point data in array
        "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
        "shape_name": str - name inherited from QuPath GeoJSON
    """
    if not isinstance(shape_data, list):
        shape_data = [shape_data]

    shapes_gj = []
    shapes_np = []
    is_points = False
    for sh in shape_data:
        if isinstance(sh, dict):
            out_shape_gj, out_shape_np = numpy_to_geojson(sh["array"], sh["shape_type"], sh["shape_name"])

        elif isinstance(sh, np.ndarray):
            out_shape_gj, out_shape_np = numpy_to_geojson(sh, **kwargs)

        else:
            if Path(sh).is_file():
                sh_fp = Path(sh)

                if sh_fp.suffix in [".json", ".geojson", ".zip"]:
                    out_shape_gj, out_shape_np, is_points = read_geojson(str(sh_fp))
                # elif sh_fp.suffix == ".cz":
                #     out_shape_gj = read_zen_shapes(str(sh_fp))
                #     out_shape_np = [gj_to_np(s) for s in out_shape_gj]
                else:
                    raise ValueError(f"{sh_fp!s} is not a geojson or numpy array")
            else:
                raise FileNotFoundError(f"{Path(sh).as_posix()!s} file not found")

        if isinstance(out_shape_gj, list):
            shapes_gj.extend(out_shape_gj)
        else:
            shapes_gj.append(out_shape_gj)

        if isinstance(out_shape_np, list):
            shapes_np.extend(out_shape_np)
        else:
            shapes_np.append(out_shape_np)
    return shapes_gj, shapes_np, is_points


def scale_shape_coordinates(poly: dict, scale_factor: float):
    """
    Scale coordinates by a factor.

    Parameters
    ----------
    poly: dict
        dict of coordinate data contain np.ndarray in "array" key
    scale_factor: float
        isotropic scaling factor for the coordinates

    Returns
    -------
    poly: dict
        dict containing coordinates scaled by scale_factor
    """
    poly_coords = poly["array"]
    poly_coords = poly_coords * scale_factor
    poly["array"] = poly_coords
    return poly


def invert_nonrigid_transforms(itk_transforms: list):
    """Check list of sequential ITK transforms for non-linear (i.e., bspline) transforms.

    Transformations need to be inverted to transform from moving to fixed space as transformations
    are mapped from fixed space to moving.
    This will first convert any non-linear transforms to a displacement field then invert the displacement field
    using ITK methods. It usually works quite well but is not an exact solution.
    Linear transforms can be inverted on  the fly when transforming points.

    Parameters
    ----------
    itk_transforms:list
        list of itk.Transform

    Returns
    -------
    itk_transforms:list
        list of itk.Transform where any non-linear transforms are replaced with an inverted displacement field
    """
    tform_linear = [t.is_linear for t in itk_transforms]

    if all(tform_linear):
        return itk_transforms
    else:
        nl_idxs = np.where(np.array(tform_linear) == 0)[0]
        for nl_idx in nl_idxs:
            if not itk_transforms[nl_idx].inverse_transform:
                print(
                    f"transform at index {nl_idx} is non-linear and the inverse has not been computed\n"
                    "inverting displacement field(s)...\n"
                    "this can take some time"
                )
                itk_transforms[nl_idx].compute_inverse_nonlinear()

    return itk_transforms


# def prepare_pt_transformation_data(transformations, compute_inverse=True):
#     """
#     Read and prepare wsireg transformation data for point set transformation
#
#     Parameters
#     ----------
#     transformations
#         list of dict containing elastix transformation data or str to wsireg .json file containing
#         elastix transformation data
#     compute_inverse : bool
#         whether or not to compute the inverse transformation for moving to fixed point transformations
#
#     Returns
#     -------
#     itk_pt_transforms:list
#         list of transformation data ready to operate on points
#     target_res:
#         physical spacing of the final transformation in the transform sequence
#         This is needed to map coordinates defined as pixel indices to physical coordinates and then back
#     """
#     if not all([isinstance(t, RegTransform) for t in transformations]):
#         _, transformations = wsireg_transforms_to_itk_composite(transformations)
#     if compute_inverse:
#         transformations = invert_nonrigid_transforms(transformations)
#     target_res = float(transformations[-1].output_spacing[0])
#     return transformations, target_res
#
#
# def itk_transform_pts(
#     pt_data: np.ndarray,
#     itk_transforms: list,
#     px_idx=True,
#     source_res=1,
#     output_idx=True,
#     target_res=2,
# ):
#     """
#     Transforms x,y points stored in np.ndarray using list of ITK transforms
#     All transforms are in physical coordinates, so all points must be converted to physical coordinates
#     before transformation, but this function allows converting back to pixel indices after transformation
#
#     Can intake points in physical coordinates is px_idx == False
#     Can output points in physical coordinates if output_idx == False
#
#     Parameters
#     ----------
#     pt_data : np.ndarray
#         array where rows are points and columns are x,y
#     itk_transforms: list
#         list of ITK transforms, non-linear transforms should be inverted
#     px_idx: bool
#         whether points are specified in physical coordinates (i.e., microns) or
#         in pixel indices
#     source_res: float
#         resolution of the image on which annotations were made
#     output_idx: bool
#         whether transformed points should be output in physical coordinates (i.e., microns) or
#         in pixel indices
#     target_res: float
#         resolution of the final target image for conversion back to pixel indices
#
#     Returns
#     -------
#     tformed_pts:np.ndarray
#         transformed points array where rows are points and columns are x,y
#
#     """
#     tformed_pts = []
#     for pt in pt_data:
#         if px_idx :
#             pt = pt * source_res
#         for idx, t in enumerate(itk_transforms):
#             if idx == 0:
#                 t_pt = t.inverse_transform.TransformPoint(pt)
#             else:
#                 t_pt = t.inverse_transform.TransformPoint(t_pt)
#         t_pt = np.array(t_pt)
#         if output_idx :
#             t_pt *= 1 / target_res
#         tformed_pts.append(t_pt)
#
#     return np.stack(tformed_pts)


def get_all_shape_coords(shapes: list):
    """Get all shape coordinates from a list of shapes."""
    return np.vstack([np.squeeze(sh["geometry"]["coordinates"][0]) for sh in shapes])


# code below is for managing transforms as masks rather than point sets
# will probably not reimplement, if segmentation data can is expressed
# as a mask, it can be transformed as an image (using attachment_modality)
def approx_polygon_contour(mask: np.ndarray, percent_arc_length=0.01):
    """
    Approximate binary mask contours to polygon vertices using cv2.

    Parameters
    ----------
    mask : numpy.ndarray
        2-d numpy array of datatype np.uint8.
    percent_arc_length : float
        scaling of epsilon for polygon approximate vertices accuracy.
        maximum distance of new vertices from original.

    Returns
    -------
    numpy.ndarray
        returns an 2d array of vertices, rows: points, columns: y,x

    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 1:
        contours = [contours[np.argmax([cnt.shape[0] for cnt in contours])]]

    epsilon = percent_arc_length * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    return np.squeeze(approx).astype(np.uint32)


# def index_mask_to_shapes(index_mask, shape_name, tf_shapes):
#     """
#     Find the polygons of a transformed shape mask, conveting binary mask
#     to list of polygon verteces and sorting by numerical index
#
#     Parameters
#     ----------
#     index_mask:np.ndarray
#         mask where each shape is defined by it's index
#     shape_name:str
#         name of the shape
#     tf_shapes:list
#         original list of shape GeoJSON data to be updated
#
#     Returns
#     -------
#     updated_shapes:list
#         dict of GeoJSON information with updated coordinate information
#     """
#     labstats = sitk.LabelShapeStatisticsImageFilter()
#     labstats.SetBackgroundValue(0)
#     labstats.Execute(index_mask)
#
#     index_mask = sitk.GetArrayFromImage(index_mask)
#     updated_shapes = deepcopy(tf_shapes)
#
#     for idx, shape in enumerate(tf_shapes):
#         if shape["properties"]["classification"]["name"] == shape_name:
#             label_bb = labstats.GetBoundingBox(idx + 1)
#             x_min = label_bb[0]
#             x_len = label_bb[2]
#             y_min = label_bb[1]
#             y_len = label_bb[3]
#
#             sub_mask = index_mask[y_min : y_min + y_len, x_min : x_min + x_len]
#
#             sub_mask[sub_mask == idx + 1] = 255
#
#             yx_coords = approx_polygon_contour(sub_mask, 0.00001)
#             xy_coords = yx_coords
#             xy_coords = np.append(xy_coords, xy_coords[:1, :], axis=0)
#             xy_coords = xy_coords + [x_min, y_min]
#             updated_shapes[idx]["geometry"]["coordinates"] = [xy_coords.tolist()]
#
#     return updated_shapes


# don't intend to maintain
# def read_zen_shapes(zen_fp):
#     """Read Zeiss Zen Blue .cz ROIs files to wsimap shapely format.
#
#     Parameters
#     ----------
#     zen_fp : str
#         file path of Zen .cz.
#
#     Returns
#     -------
#     list
#         list of wsimap shapely rois
#
#     """
#
#     root = etree.parse(zen_fp)
#
#     rois = root.xpath("//Elements")[0]
#     shapes_out = []
#     for shape in rois:
#         try:
#             ptset_name = shape.find("Attributes/Name").text
#         except AttributeError:
#             ptset_name = "unnamed"
#
#         if shape.tag == "Polygon":
#             ptset_cz = shape.find("Geometry/Points")
#             # ptset_type = "Polygon"
#
#             poly_str = ptset_cz.text
#             poly_str = poly_str.split(" ")
#             poly_str = [poly.split(",") for poly in poly_str]
#             poly_str = [[float(pt[0]), float(pt[1])] for pt in poly_str]
#
#             poly = {
#                 "geometry": geojson.Polygon(poly_str),
#                 "properties": {"classification": {"name": ptset_name}},
#             }
#
#             shapes_out.append(poly)
#
#         if shape.tag == "Rectangle":
#             rect_pts = shape.find("Geometry")
#
#             x = float(rect_pts.findtext("Left"))
#             y = float(rect_pts.findtext("Top"))
#             width = float(rect_pts.findtext("Width"))
#             height = float(rect_pts.findtext("Height"))
#
#             rect = geojson.Polygon(
#                 [
#                     [x, y],
#                     [x + width, y],
#                     [x + width, y + height],
#                     [x, y + height],
#                     [x, y],
#                 ]
#             )
#
#             rect = {
#                 "geometry": rect,
#                 "properties": {"classification": {"name": ptset_name}},
#             }
#
#             shapes_out.append(rect)
#
#     return shapes_out
