from colormath.color_objects import SpectralColor,LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976,delta_e_cie2000


# 将400-700范围内的reflectance转化为SpectralColor对象
def reflectance_to_spectral_color(reflectance, observer='10', illuminant='d65', start=400, end=710):
    spc = SpectralColor(
        observer=observer, illuminant=illuminant,
        spec_400nm=reflectance[0], spec_410nm=reflectance[1], spec_420nm=reflectance[2],
        spec_430nm=reflectance[3], spec_440nm=reflectance[4], spec_450nm=reflectance[5],
        spec_460nm=reflectance[6], spec_470nm=reflectance[7], spec_480nm=reflectance[8],
        spec_490nm=reflectance[9], spec_500nm=reflectance[10], spec_510nm=reflectance[11],
        spec_520nm=reflectance[12], spec_530nm=reflectance[13], spec_540nm=reflectance[14],
        spec_550nm=reflectance[15], spec_560nm=reflectance[16], spec_570nm=reflectance[17],
        spec_580nm=reflectance[18], spec_590nm=reflectance[19], spec_600nm=reflectance[20],
        spec_610nm=reflectance[21], spec_620nm=reflectance[22], spec_630nm=reflectance[23],
        spec_640nm=reflectance[24], spec_650nm=reflectance[25], spec_660nm=reflectance[26],
        spec_670nm=reflectance[27], spec_680nm=reflectance[28], spec_690nm=reflectance[29],
        spec_700nm=reflectance[30])
    return spc


def reflectance2lab(reflectance):
    spec = reflectance_to_spectral_color(reflectance)
    lab = convert_color(spec, LabColor)
    return lab


# 计算CIE1976色差
def cie1976_color_diff(reflectance1, reflectance2):
    lab1 = reflectance2lab(reflectance1)
    lab2 = reflectance2lab(reflectance2)
    color_diff = delta_e_cie1976(lab1, lab2)
    return color_diff


# 计算CIEDE2000色差(kL、kC、kH取1)
def ciede2000_color_diff(reflectance1, reflectance2):
    lab1 = reflectance2lab(reflectance1)
    lab2 = reflectance2lab(reflectance2)
    color_diff = delta_e_cie2000(lab1, lab2)
    return color_diff
