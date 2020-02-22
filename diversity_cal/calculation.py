import numpy as np
from utils.color_utils import ciede2000_color_diff, conc2ref_km
from itertools import combinations


def main():
    # 工厂给的配方和对应的分光反射率数据
    factory_formula = np.load('factory_formula.npy')
    factory_reflectance = np.load('factory_reflectance.npy')
    # 计算分光反射率
    km_ref = conc2ref_km(factory_formula)
    # 选取实验样本
    sample_conc = factory_formula[0]
    sample_ref = km_ref[0]

    # 选择1种、2种，3种色浆的排列组合方式
    combine_list1 = list(combinations(np.arange(0, 21, 1), 20))
    combine_list2 = list(combinations(np.arange(0, 21, 1), 19))
    combine_list3 = list(combinations(np.arange(0, 21, 1), 18))
    result1 = combine(combine_list1, sample_ref)
    result2 = combine(combine_list2, sample_ref)
    result3 = combine(combine_list3, sample_ref)
    return result1 + result2 + result3


def combine(combine_list, sample_ref):
    combine_num = combine_list.__len__()
    combine_array = np.ones(combine_num * 21).reshape(combine_num, 21)
    for i in range(combine_num):
        combine_array[i, combine_list[i]] = 0.
    print(combine_array)
    return 0


main()
