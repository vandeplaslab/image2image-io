"""Tests for GeoJSON utilities."""

import json
import zipfile

import numpy as np
import pytest

from image2image_io.readers.geojson_utils import (
    _parse_geojson_data,
    add_unnamed,
    geojson_to_numpy,
    get_all_shape_coords,
    get_int_dtype,
    numpy_to_geojson,
    read_geojson,
    shape_reader,
)


def test_parse_geojson_data_mixed_features_preserves_point_flag():
    gj_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                "properties": {},
            },
        ],
    }

    geojson_data, shapes_np, is_points = _parse_geojson_data(gj_data)

    assert is_points
    assert len(geojson_data) == 2
    assert len(shapes_np) == 2
    assert shapes_np[0]["shape_type"] == "Point"


def test_parse_geojson_data_adds_unnamed_properties():
    gj_data = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}}]

    geojson_data, _shapes_np, _is_points = _parse_geojson_data(gj_data)

    assert geojson_data[0]["properties"]["classification"]["name"] == "unnamed"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, np.uint8), (255, np.uint8), (256, np.uint16), (65536, np.int32)],
)
def test_get_int_dtype(value, expected):
    assert get_int_dtype(value) == expected


def test_get_int_dtype_raises_for_large_value():
    with pytest.raises(ValueError, match="Too many shapes"):
        get_int_dtype(np.iinfo(np.uint32).max + 1)


def test_geojson_to_numpy_handles_multipoint_and_linestring():
    multipoint = {
        "type": "Feature",
        "geometry": {"type": "MultiPoint", "coordinates": [[1, 2], [3, 4]]},
        "properties": {"classification": {"name": "pts"}},
    }
    linestring = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]},
        "properties": {"classification": {"name": "line"}},
    }

    point_data, is_points = geojson_to_numpy(multipoint)
    line_data, line_is_points = geojson_to_numpy(linestring)

    assert is_points
    assert not line_is_points
    np.testing.assert_array_equal(point_data[0]["array"], np.array([[1, 2], [3, 4]], dtype=np.float32))
    np.testing.assert_array_equal(line_data[0]["array"], np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_geojson_to_numpy_rejects_unsupported_geometry():
    feature = {"type": "Feature", "geometry": {"type": "GeometryCollection", "coordinates": []}, "properties": {}}

    with pytest.raises(ValueError, match="not supported"):
        geojson_to_numpy(feature)


def test_add_unnamed_preserves_existing_classification():
    feature = {"properties": {"classification": {"name": "kept"}}}
    assert add_unnamed(feature)["properties"]["classification"]["name"] == "kept"


def test_read_geojson_from_json_and_zip(tmp_path):
    gj_data = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {}}],
    }
    json_path = tmp_path / "shapes.geojson"
    zip_path = tmp_path / "shapes.zip"
    json_path.write_text(json.dumps(gj_data))
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("shapes.geojson", json.dumps(gj_data))

    json_geojson, json_shapes, json_is_points = read_geojson(json_path)
    zip_geojson, zip_shapes, zip_is_points = read_geojson(zip_path)

    assert json_is_points and zip_is_points
    assert len(json_geojson) == len(zip_geojson) == 1
    np.testing.assert_array_equal(json_shapes[0]["array"], zip_shapes[0]["array"])


def test_numpy_to_geojson_polygon_and_linestring():
    polygon = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    line = np.array([[0, 1], [2, 3]], dtype=np.float32)

    polygon_gj, polygon_np = numpy_to_geojson(polygon, shape_type="polygon", shape_name="poly")
    line_gj, line_np = numpy_to_geojson(line, shape_type="linestring", shape_name="line")

    assert polygon_gj["geometry"]["type"] == "Polygon"
    assert polygon_np["array"].shape[0] == 4
    assert line_gj["geometry"]["type"] == "LineString"
    assert line_np["shape_name"] == "line"


def test_shape_reader_supports_dicts_arrays_and_files(tmp_path):
    array = np.array([[0, 0], [1, 1]], dtype=np.float32)
    geojson_path = tmp_path / "shape.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {}}]})
    )

    shapes_gj, shapes_np, is_points = shape_reader(
        [{"array": array, "shape_type": "polygon", "shape_name": "roi"}, array, str(geojson_path)],
        shape_type="linestring",
        shape_name="line",
    )

    assert len(shapes_gj) == 3
    assert len(shapes_np) == 3
    assert is_points


def test_shape_reader_raises_for_missing_or_invalid_files(tmp_path):
    missing = tmp_path / "missing.geojson"
    invalid = tmp_path / "bad.txt"
    invalid.write_text("bad")

    with pytest.raises(FileNotFoundError):
        shape_reader(str(missing))
    with pytest.raises(ValueError, match="not a geojson or numpy array"):
        shape_reader(str(invalid))


def test_get_all_shape_coords_stacks_polygon_coordinates():
    shapes = [
        {"geometry": {"coordinates": [[[0, 0], [1, 0], [0, 0]]]}},
        {"geometry": {"coordinates": [[[2, 2], [3, 2], [2, 2]]]}}
    ]

    coords = get_all_shape_coords(shapes)

    np.testing.assert_array_equal(coords, np.array([[0, 0], [1, 0], [0, 0], [2, 2], [3, 2], [2, 2]]))
