"""
测试每一个波长下增长0.01对最终的色差的影响
测试结果说明0.01级别的改变，对色差的影响还是比较大的
假设波长A,B增加，波长C,D减少
最终的色差变化为A,B变化之和E1与C,D变化之和E2的差的绝对值
"""
import numpy as np
import copy
from utils.color_utils import conc2ref_km, ciede2000_color_diff

factory_formula = np.load('factory_formula.npy')
factory_ref = np.load('factory_reflectance.npy')
cal_ref = conc2ref_km(factory_formula)
for i in range(factory_ref.shape[0]):
    diff = ciede2000_color_diff(factory_ref[i], cal_ref[i])
for i in range(28):
    ref1 = copy.copy(cal_ref[0])
    ref2 = copy.copy(cal_ref[0])
    ref1[i] += 0.01
    diff = ciede2000_color_diff(ref1, ref2)
    print(diff)
