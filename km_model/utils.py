import torch
from colormath.color_objects import SpectralColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2. * XY)

def fit(input, target):
    return torch.mean((input - target) ** 2)

def non_nagative_attachment(base, lamb, x):
    return 1. / torch.clamp(torch.pow(base, lamb * x[x < 0]), min=0.001)