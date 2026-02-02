"""Transformer."""

from __future__ import annotations
import SimpleITK as sitk
import numpy as np
from copy import deepcopy
import math


class Transformer:
    """Transformer functions for image processing using SimpleITK."""

    def __init__(
        self,
        flip_h: bool = False,
        flip_v: bool = False,
        angle: float = 0,
        output_size: tuple[int, int] | None = None,
        output_spacing: tuple[float, float] | None = None,
    ):
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.angle = angle
        self.output_size = output_size
        self.output_spacing = output_spacing

    @classmethod
    def from_reader(cls, reader, flip_h: bool = False, flip_v: bool = False, angle: float = 0):
        """From reader."""
        obj = cls(flip_h=flip_h, flip_v=flip_v, angle=angle)
        image = obj._convert_image(reader.get_channel(0, 0), resolution=reader.resolution)
        params = obj.get_parameters(image)[0]
        obj.output_spacing = tuple(float(s) for s in params["Spacing"])
        obj.output_size = tuple(int(s) for s in params["Size"])
        return obj

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Apply transformation to the image."""
        return self._resample(image)

    def _convert_image(self, image: np.ndarray | sitk.Image, resolution: float) -> sitk.Image:
        """Convert numpy array to SimpleITK image if necessary."""
        if not isinstance(image, sitk.Image):
            image = sitk.GetImageFromArray(image)
            image.SetSpacing((resolution, resolution))
        return image

    def _resample(self, image: sitk.Image) -> sitk.Image:
        """Resample the image to a new size while preserving spatial metadata."""
        transform = self.get_transform(image)

        interpolator = sitk.sitkNearestNeighbor
        resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
        resampler.SetOutputOrigin(image.GetOrigin())  # type: ignore[no-untyped-call]
        # resampler.SetOutputDirection(self.output_direction)  # type: ignore[no-untyped-call]
        resampler.SetSize(image.GetSize())  # type: ignore[no-untyped-call]
        resampler.SetOutputSpacing(image.GetSpacing())  # type: ignore[no-untyped-call]
        resampler.SetInterpolator(interpolator)  # type: ignore[no-untyped-call]
        resampler.SetTransform(transform)  # type: ignore[no-untyped-call]
        return resampler.Execute(image)  # type: ignore[no-untyped-call]

    def get_transform(self, image: sitk.Image):
        """Transform."""
        composite_transform = sitk.CompositeTransform(2)
        if self.flip_h:
            composite_transform.AddTransform(
                convert_to_itk(generate_affine_flip_transform(image, image.GetSpacing()[0], flip="h"))
            )
        if self.flip_v:
            composite_transform.AddTransform(
                convert_to_itk(generate_affine_flip_transform(image, image.GetSpacing()[0], flip="v"))
            )
        if self.angle:
            composite_transform.AddTransform(
                convert_to_itk(generate_rigid_rotation_transform(image, image.GetSpacing()[0], self.angle))
            )
        return composite_transform

    def get_parameters(self, image: sitk.Image) -> list[dict]:
        """Transform."""
        composite_transform = []
        if self.flip_h:
            composite_transform.append(generate_affine_flip_transform(image, image.GetSpacing()[0], flip="h"))
        if self.flip_v:
            composite_transform.append(generate_affine_flip_transform(image, image.GetSpacing()[0], flip="v"))
        if self.angle:
            composite_transform.append(generate_rigid_rotation_transform(image, image.GetSpacing()[0], self.angle))
        return composite_transform


BASE_RIGID_TRANSFORM = {
    "Transform": ["EulerTransform"],
    "NumberOfParameters": ["3"],
    "TransformParameters": ["0", "0", "0"],  # ?, tx, ty
    "InitialTransformParametersFileName": ["NoInitialTransform"],
    "HowToCombineTransforms": ["Compose"],
    "FixedImageDimension": ["2"],
    "MovingImageDimension": ["2"],
    "FixedInternalImagePixelType": ["float"],
    "MovingInternalImagePixelType": ["float"],
    "Size": ["0", "0"],
    "Index": ["0", "0"],
    "Spacing": ["", ""],
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
BASE_AFFINE_TRANSFORM = {
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


def compute_rotation_bounds(shape: tuple[int, int], angle_deg: float = 0) -> tuple[float, float]:
    """Compute rotation bounds."""
    h, w = shape
    theta = np.radians(angle_deg)
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    bound_w = (h * s) + (w * c)
    bound_h = (h * c) + (w * s)
    return bound_w, bound_h


def compute_rotation_bounds_for_image(image: sitk.Image, angle_deg: float = 0) -> tuple[float, float]:
    """Compute the bounds of an image after by an angle.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated angle
    angle_deg : float
        angle of rotation in degrees, rotates counter-clockwise if positive

    Returns
    -------
    tuple of the rotated image's size in x and y

    """
    w, h = image.GetSize()[0:2]  # type: ignore[no-untyped-call]
    return compute_rotation_bounds((h, w), angle_deg=angle_deg)


def generate_affine_flip_transform(image: sitk.Image, spacing: float, flip: str = "h") -> dict:
    """Generate a SimpleElastix transformation parameter Map to horizontally or vertically flip image.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    flip : str
        "h" or "v" for horizontal or vertical flipping, respectively

    Returns
    -------
    SimpleITK.ParameterMap of flipping transformation (AffineTransform)

    """
    tform = deepcopy(BASE_AFFINE_TRANSFORM)
    image.SetSpacing((spacing, spacing))
    bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)
    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(((bound_w - 1) / 2, (bound_h - 1) / 2))

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(int(bound_w)), str(int(bound_h))]

    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform_params = ["1", "0", "0", "1", "0", "0"]
    if flip == "h":
        tform_params = ["-1", "0", "0", "1", "0", "0"]
    elif flip == "v":
        tform_params = ["1", "0", "0", "-1", "0", "0"]
    tform["TransformParameters"] = tform_params
    return tform


def generate_rigid_rotation_transform(image: sitk.Image, spacing: float, angle_deg: float) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    angle_deg : float
        angle of rotation in degrees, rotates counter-clockwise if positive.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w_px, bound_h_px = compute_rotation_bounds_for_image(image, angle_deg=angle_deg)
    # calculate rotation center point
    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w_px - 1) / 2, (bound_h_px - 1) / 2),
    )  # type: ignore[no-untyped-call]

    size = image.GetSize()
    c_x, c_y = (size[0] - 1) / 2, (size[1] - 1) / 2  # type: ignore[no-untyped-call]
    c_x_phy, c_y_phy = image.TransformContinuousIndexToPhysicalPoint((c_x, c_y))  # type: ignore[no-untyped-call]
    translation_x_phy = rot_x_phy - c_x_phy
    translation_y_phy = rot_y_phy - c_y_phy

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(math.ceil(bound_w_px)), str(math.ceil(bound_h_px))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [
        str(np.radians(angle_deg)),
        str(-1 * translation_x_phy),
        str(-1 * translation_y_phy),
    ]
    return tform


def convert_to_itk(tform: dict) -> sitk.Transform:
    """Convert Elastix transform to ITK transform."""
    itk_tform: sitk.Euler2DTransform | sitk.Similarity2DTransform | sitk.AffineTransform
    if tform["Transform"][0] == "EulerTransform":
        itk_tform = euler_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "SimilarityTransform":
        itk_tform = similarity_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "AffineTransform":
        itk_tform = affine_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "TranslationTransform":
        itk_tform = euler_elx_to_itk2d(tform, is_translation=True)
    else:
        raise ValueError(f"Transform {tform['Transform'][0]} not supported")
    return itk_tform


def euler_elx_to_itk2d(tform: dict, is_translation: bool = False) -> sitk.Euler2DTransform:
    """Convert Elastix Euler transform to ITK Euler transform."""
    euler2d = sitk.Euler2DTransform()
    if is_translation:
        elx_parameters = [0]
        elx_parameters_trans = [float(p) for p in tform["TransformParameters"]]
        elx_parameters.extend(elx_parameters_trans)
    else:
        center = [float(p) for p in tform["CenterOfRotationPoint"]]
        euler2d.SetFixedParameters(center)
        elx_parameters = [float(p) for p in tform["TransformParameters"]]
    euler2d.SetParameters(elx_parameters)
    return euler2d


def similarity_elx_to_itk2d(tform: dict) -> sitk.Similarity2DTransform:
    """Convert Elastix similarity transform to ITK similarity transform."""
    similarity2d = sitk.Similarity2DTransform()

    center = [float(p) for p in tform["CenterOfRotationPoint"]]
    similarity2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform["TransformParameters"]]
    similarity2d.SetParameters(elx_parameters)
    return similarity2d


def affine_elx_to_itk2d(tform: dict) -> sitk.AffineTransform:
    """Convert Elastix affine transform to ITK affine transform."""
    im_dimension = len(tform["Size"])
    affine2d = sitk.AffineTransform(im_dimension)

    center = [float(p) for p in tform["CenterOfRotationPoint"]]
    affine2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform["TransformParameters"]]
    affine2d.SetParameters(elx_parameters)
    return affine2d
