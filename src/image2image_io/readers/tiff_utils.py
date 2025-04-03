"""TIFF file utilities.

Taken from:
https://github.com/NHPatterson/napari-imsmicrolink/blob/master/src/napari_imsmicrolink/utils/tifffile_meta.py
"""

from __future__ import annotations

from ome_types.model import OME
from tifffile import TiffFile


def qptiff_channel_names(tif: TiffFile, series_index: int = 0) -> list[str]:
    """Retrieve filenames from qptiff."""
    from image2image_io.utils.utilities import xmlstr_to_dict

    channel_names: list[str] = []
    pages = tif.series[series_index].pages
    if not pages:
        return channel_names
    for index, page in enumerate(pages):
        xml = page.tags["ImageDescription"].value
        xml_dict = xmlstr_to_dict(xml)
        if "PerkinElmer-QPI-ImageDescription" in xml_dict:
            xml_dict = xml_dict["PerkinElmer-QPI-ImageDescription"]
            channel_name = []
            if "Biomarker" in xml_dict:
                channel_name.append(xml_dict["Biomarker"])
            if "Name" in xml_dict:
                channel_name.append(xml_dict["Name"])
            channel_name = " - ".join(channel_name)  # type: ignore[assignment]
            channel_name += f" (C{index})"
            channel_names.append(channel_name)  # type: ignore[arg-type]
    return channel_names


def tifftag_xy_pixel_sizes(rdr: TiffFile, series_idx: int, level_idx: int) -> tuple[float, ...]:
    """Resolution data is stored in the TIFF tags in pixels per cm, this is converted to microns per pixel."""
    # pages are accessed because they contain the tiff tag
    # subset by series -> level -> first page contains all tags
    current_page = rdr.series[series_idx].levels[level_idx].pages[0]

    x_res = current_page.tags["XResolution"].value
    y_res = current_page.tags["XResolution"].value

    res_unit = current_page.tags["ResolutionUnit"].value

    # convert units to micron
    # res_unit == 1: undefined (px)
    # res_unit == 2 pixels per inch
    # res unit == 3 pixels per cm
    # in all cases we convert to um
    # https://www.awaresystems.be/imaging/tiff/tifftags/resolutionunit.html
    if res_unit.value == 1:
        res_to_um = 1
    if res_unit.value == 2:
        res_to_um = 25400
    elif res_unit.value == 3:
        res_to_um = 10000

    # conversion of pixels / um to um / pixel
    x_res_um = (1 / (x_res[0] / x_res[1])) * res_to_um
    y_res_um = (1 / (y_res[0] / y_res[1])) * res_to_um
    return x_res_um, y_res_um


def svs_xy_pixel_sizes(rdr: TiffFile, series_idx: int, level_idx: int) -> tuple[float, ...]:
    """Get resolution data stored in ImageDescription of SVS."""
    # pages are accessed because they contain the tiff tag
    # subset by series -> level -> first page contains all tags
    current_page = rdr.series[series_idx].levels[level_idx].pages[0]
    id_str = current_page.tags["ImageDescription"].value
    svs_info = id_str.split("|")
    mpp_val = None
    for i in svs_info:
        if "MPP" in i:
            mpp_val = float(i.split("=")[1])

    if mpp_val:
        return mpp_val, mpp_val
    return 1.0, 1.0


def ometiff_xy_pixel_sizes(ome_metadata: OME, series_idx: int):
    """Get resolution data stored in OME metadata."""
    from pint import UnitRegistry

    ps_x = ome_metadata.images[series_idx].pixels.physical_size_x
    ps_y = ome_metadata.images[series_idx].pixels.physical_size_y
    ps_unit = ome_metadata.images[series_idx].pixels.physical_size_x_unit

    if ps_unit.name.lower() != "micrometer":
        ureg = UnitRegistry()
        cur_size = ps_x * ureg(ps_unit.name.lower())
        out_um = cur_size.to("micrometer")
        ps_x_out = out_um.magnitude
        return (ps_x_out, ps_x_out)
    else:
        return (ps_x, ps_y)


def ometiff_ch_names(ome_metadata: OME, series_idx: int):
    """Get channel names stored in OME metadata."""
    ch_meta = ome_metadata.images[series_idx].pixels.channels
    cnames = []
    for idx, ch in enumerate(ch_meta):
        if ch.name:
            cnames.append(ch.name)
        else:
            cnames.append(f"C{str(idx + 1).zfill(2)}")

    return cnames
