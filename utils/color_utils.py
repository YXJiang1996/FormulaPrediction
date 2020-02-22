import numpy as np
import utils.base_info as info
from colormath.color_objects import SpectralColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000


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


# 将分光反射率转化为lab颜色
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


# 使用km模型计算配方的分光反射率
# 注意：这里的concentrations使用的是 color_num*sample_num 的二维数组
def conc2ref_km(concentrations, background=info.white_solvent_reflectance,
                base_conc=info.base_concentration,
                base_color_num=info.base_color_num, base_ref=info.base_reflectance,
                ref_dim=info.reflectance_dim):
    init_conc_array = np.repeat(base_conc.reshape(base_color_num, 1), ref_dim).reshape(base_color_num, ref_dim)
    reflectance = np.zeros(ref_dim * concentrations.shape[0]).reshape(ref_dim, concentrations.shape[0])

    fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
    fst = ((np.ones_like(base_ref) - base_ref) ** 2 / (base_ref * 2) - fsb) / init_conc_array
    fss = np.zeros(31 * concentrations.shape[0]).reshape(31, concentrations.shape[0])
    for i in range(info.reflectance_dim):
        for j in range(info.base_color_num):
            fss[i, :] += concentrations[:, j] * fst[j, i]
        fss[i, :] += np.ones(concentrations.shape[0]) * fsb[i]

    reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
    reflectance = reflectance.transpose()
    return reflectance
