"""ITK utilities."""

from copy import deepcopy

import numpy as np

AFFINE_TRANSFORM = {
    "Transform": ["AffineTransform"],
    "NumberOfParameters": ["6"],
    "TransformParameters": ["1", "0", "0", "1", "0", "0"],
    "InitialTransformParametersFileName": ["NoInitialTransform"],
    "HowToCombineTransforms": ["Compose"],
    "FixedImageDimension": ["2"],
    "MovingImageDimension": ["2"],
    "FixedInternalImagePixelType": ["float"],
    "MovingInternalImagePixelType": ["float"],
    "Size": ["0", "0"],
    "Index": ["0", "0"],
    "Spacing": ["0", "0"],
    "Origin": ["0.0000", "0.0000"],
    "Direction": [
        "1.0000000000",
        "0.0000000000",
        "0.0000000000",
        "1.0000000000",
    ],
    "UseDirectionCosines": ["true"],
    "CenterOfRotationPoint": ["0", "0"],
    "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
    "Resampler": ["DefaultResampler"],
    "DefaultPixelValue": ["0.000000"],
    "ResultImageFormat": ["mha"],
    "ResultImagePixelType": ["float"],
    "CompressResultImage": ["true"],
}


def compute_affine_bound(
    shape: tuple[int, int], affine: np.ndarray, spacing: float = 1
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute affine bounds."""
    w, h = shape
    # Top-left, Top-right, Bottom-left, Bottom-right
    corners = np.array(
        [
            [0, 0, 1],  # Adding 1 for homogeneous coordinates (x, y, 1)
            [w, 0, 1],
            [0, h, 1],
            [w, h, 1],
        ]
    )
    # Apply affine transformation to corners
    transformed_corners = np.dot(corners, affine.T)  # Transpose matrix to match shapes

    # Extracting x and y coordinates
    x_coords, y_coords = transformed_corners[:, 0], transformed_corners[:, 1]

    # Calculate bounds
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Calculate new  width and height
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    affine_ = affine[:2, :]
    new_width += abs(affine_[1, 2] / spacing)
    new_height += abs(affine_[0, 2] / spacing)
    return (new_width, new_height), (new_width / 2, new_height / 2)


def affine_to_itk_affine(
    affine: np.ndarray,
    image_shape: tuple[int, int],
    spacing: float = 1.0,
    inverse: bool = False,
) -> dict:
    """Convert affine matrix (yx, um) to ITK affine matrix.

    The assumption is that the affine matrix is provided in numpy ordering (e.g. from napari) and values are in um.
    """
    # TODO change the origin so that we don't have to make the image bigger

    assert affine.shape == (3, 3), "affine matrix must be 3x3"
    if inverse:
        affine = np.linalg.inv(affine)

    tform = deepcopy(AFFINE_TRANSFORM)
    tform["Spacing"] = [str(spacing), str(spacing)]
    # compute new image shape
    (bound_w, bound_h), (origin_x, origin_y) = compute_affine_bound(image_shape, affine, spacing)  # width, height

    # calculate rotation center point
    # center_of_rot = calculate_center_of_rotation(affine, image_shape, (spacing, spacing))
    # center_of_rot = image.TransformContinuousIndexToPhysicalPoint(
    #     ((bound_w - 1) / 2, (bound_h - 1) / 2),
    # )  # type: ignore[no-untyped-call]
    # center_of_rot = ((bound_w - 1) / 2, (bound_h - 1) / 2)
    # tform["CenterOfRotationPoint"] = [str(center_of_rot[0]), str(center_of_rot[1])]

    # adjust for pixel spacing
    tform["Size"] = [str(int(np.ceil(bound_w))), str(int(np.ceil(bound_h)))]
    # tform["Origin"] = [str(origin_x), str(origin_y)]

    # extract affine parameters
    affine_ = affine[:2, :]
    # a, b, c, d, tx, ty
    tform["TransformParameters"] = [
        affine_[1, 1],
        affine_[1, 0],
        affine_[0, 1],
        affine_[0, 0],
        affine_[1, 2],
        affine_[0, 2],
    ]
    return tform
