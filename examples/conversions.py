# -*- coding: utf-8 -*-
"""
This module shows you how to perform color space conversions. Please see the
chart on www.brucelindbloom.com/Math.html for an illustration of the conversions
you may perform.
"""

# Does some sys.path manipulation so we can run examples in-place.
# noinspection PyUnresolvedReferences
import example_config

from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, LCHabColor, SpectralColor, sRGBColor, \
    XYZColor, LCHuvColor, IPTColor


def example_lab_to_xyz():
    """
    This function shows a simple conversion of an Lab color to an XYZ color.
    """

    print("=== Simple Example: Lab->XYZ ===")
    # Instantiate an Lab color object with the given values.
    lab = LabColor(0.903, 16.296, -2.22)
    # Show a string representation.
    print(lab)
    # Convert to XYZ.
    xyz = convert_color(lab, XYZColor)
    print(xyz)
    print("=== End Example ===\n")


def example_lchab_to_lchuv():
    """
    This function shows very complex chain of conversions in action.

    LCHab to LCHuv involves four different calculations, making this the
    conversion requiring the most steps.
    """

    print("=== Complex Example: LCHab->LCHuv ===")
    # Instantiate an LCHab color object with the given values.
    lchab = LCHabColor(0.903, 16.447, 352.252)
    # Show a string representation.
    print(lchab)
    # Convert to LCHuv.
    lchuv = convert_color(lchab, LCHuvColor)
    print(lchuv)
    print("=== End Example ===\n")


def example_lab_to_rgb():
    """
    Conversions to RGB are a little more complex mathematically. There are also
    several kinds of RGB color spaces. When converting from a device-independent
    color space to RGB, sRGB is assumed unless otherwise specified with the
    target_rgb keyword arg.
    """

    print("=== RGB Example: Lab->RGB ===")
    # Instantiate an Lab color object with the given values.
    lab = LabColor(0.903, 16.296, -2.217)
    # Show a string representation.
    print(lab)
    # Convert to XYZ.
    rgb = convert_color(lab, sRGBColor)
    print(rgb)
    print("=== End Example ===\n")


def example_rgb_to_xyz():
    """
    The reverse is similar.
    """

    print("=== RGB Example: RGB->XYZ ===")
    # Instantiate an Lab color object with the given values.
    rgb = sRGBColor(120, 130, 140)
    # Show a string representation.
    print(rgb)
    # Convert RGB to XYZ using a D50 illuminant.
    xyz = convert_color(rgb, XYZColor, target_illuminant='D50')
    print(xyz)
    print("=== End Example ===\n")


def example_spectral_to_xyz():
    """
    Instantiate an Lab color object with the given values. Note that the
    spectral range can run from 340nm to 830nm. Any omitted values assume a
    value of 0.0, which is more or less ignored. For the distribution below,
    we are providing an example reading from an X-Rite i1 Pro, which only
    measures between 380nm and 730nm.
    """

    print("=== Example: Spectral->XYZ ===")
    spc = SpectralColor(
        observer='2', illuminant='d50',
        spec_380nm=0.0600, spec_390nm=0.0600, spec_400nm=0.0641,
        spec_410nm=0.0654, spec_420nm=0.0645, spec_430nm=0.0605,
        spec_440nm=0.0562, spec_450nm=0.0543, spec_460nm=0.0537,
        spec_470nm=0.0541, spec_480nm=0.0559, spec_490nm=0.0603,
        spec_500nm=0.0651, spec_510nm=0.0680, spec_520nm=0.0705,
        spec_530nm=0.0736, spec_540nm=0.0772, spec_550nm=0.0809,
        spec_560nm=0.0870, spec_570nm=0.0990, spec_580nm=0.1128,
        spec_590nm=0.1251, spec_600nm=0.1360, spec_610nm=0.1439,
        spec_620nm=0.1511, spec_630nm=0.1590, spec_640nm=0.1688,
        spec_650nm=0.1828, spec_660nm=0.1996, spec_670nm=0.2187,
        spec_680nm=0.2397, spec_690nm=0.2618, spec_700nm=0.2852,
        spec_710nm=0.2500, spec_720nm=0.2400, spec_730nm=0.2300)
    xyz = convert_color(spc, XYZColor)
    print(xyz)
    print("=== End Example ===\n")


def example_lab_to_ipt():
    """
    This function shows a simple conversion of an XYZ color to an IPT color.
    """

    print("=== Simple Example: XYZ->IPT ===")
    # Instantiate an XYZ color object with the given values.
    xyz = XYZColor(0.5, 0.5, 0.5, illuminant='d65')
    # Show a string representation.
    print(xyz)
    # Convert to IPT.
    ipt = convert_color(xyz, IPTColor)
    print(ipt)
    print("=== End Example ===\n")

# Feel free to comment/un-comment examples as you please.
example_lab_to_xyz()
example_lchab_to_lchuv()
example_lab_to_rgb()
example_spectral_to_xyz()
example_rgb_to_xyz()
example_lab_to_ipt()
