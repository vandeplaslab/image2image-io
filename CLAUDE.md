# image2image-io
Gives one reader/writer interface to TIFF/OME-TIFF, CZI, OME-Zarr, H5AD, IMS, shapes and point data, with per-channel and pyramid access.

## Role in ecosystem
IO foundation for the image2image family.
Internal deps: imzy, koyo (imported, undeclared: image2image-reg)
Depended on by: image2image, image2image-reg, imspy-data, ionglow (imported, undeclared: imspy-core)

## Key abstractions
- `readers.get_reader` / `get_simple_reader` — open any supported file
- `readers/_base_reader.BaseReader` → `TiffImageReader`, `CziImageReader`, `OmeZarrImageReader`, `ArrayImageReader`, `CoordinateImageReader`, `H5ADReader`, `ShapesReader`, `PointsReader`
- `wrapper.ImageWrapper` — collection of readers
- `writers/` — `OmeTiffWriter`, `MergeOmeTiffWriter`, `OmeZarrWrapper`
- `utils/transformer.Transformer`, `utils/warp.ImageWarper`, `models/transform.TransformData`
- `volume.Volume`; `cli/` (convert, crop, merge, thumbnail, transform)

## Do not re-explain
When working in this repo, assume familiarity with: image2image-reg, imzy, koyo
