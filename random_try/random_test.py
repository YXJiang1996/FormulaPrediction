import numpy as np
import random
import sys

sys.path.append('..')
from time import time
from tqdm import tqdm
from itertools import combinations

import km_model.info as info
from km_model.utils import conc2ref_km, ciede2000_color_diff, plot_xy


def random_test(test_conc, test_ref, sample_num):
    t_start = time()
    min_color_diff = 10000
    best_formula = []
    for i in range(sample_num):
        gen_conc, gen_ref = generate_random()
        color_diff = ciede2000_color_diff(test_ref, gen_ref)
        min_color_diff = min_color_diff if min_color_diff < color_diff else color_diff
        best_formula = best_formula if min_color_diff < color_diff else gen_conc
    time_cost = time() - t_start
    return best_formula, min_color_diff, time_cost


# 生成一个随机的配方和它对应的分光反射率
def generate_random():
    # 生成21种色浆的排列组合
    combine_list = list(combinations(np.arange(0, info.base_color_num, 1), info.base_color_num - info.chosen_color_num))
    # 随机生成一个配方
    test_conc = np.random.uniform(0, 1, size=(1, info.base_color_num))
    test_combine = list(random.choice(combine_list))
    test_conc[0, test_combine] = 0.
    test_ref = conc2ref_km(test_conc)
    return test_conc[0], test_ref[0]


def main():
    # 生成一组配方和对应的分光反射率
    test_conc, test_ref = generate_random()
    # 测试参数
    sample_num_arr = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18,
                      2 ** 19, 2 ** 20]  # 样本数目
    n_average_diff_arr = []  # 色差数组
    n_min_diff_arr = []
    n_max_diff_arr = []
    time_cost_arr = []  # 时间数组
    for num_index in range(sample_num_arr.__len__()):
        sample_num = sample_num_arr[num_index]
        n_total_diff = 0  # n次测试得到的色差之和
        n_total_time = 0  # n次测试花费的总时间
        n_max_diff = 0  # n次测试得到的最小色差中最大的
        n_min_diff = 100  # n次测试得到的最小色差中最小的
        repeat_times = 10  # 对于同一采样数重复试验次数
        print('sample_num:', sample_num)
        for i in range(repeat_times):
            best_formula, min_diff, time_cost = random_test(test_conc, test_ref, sample_num)
            n_total_diff = n_total_diff + min_diff
            n_total_time = n_total_time + time_cost
            n_min_diff = min_diff if min_diff < n_min_diff else n_min_diff
            n_max_diff = min_diff if min_diff > n_max_diff else n_max_diff
        average_time = n_total_time / repeat_times
        average_diff = n_total_diff / repeat_times
        n_average_diff_arr.append(average_diff)
        n_min_diff_arr.append(n_min_diff)
        n_max_diff_arr.append(n_max_diff)
        time_cost_arr.append(average_time)
        print('10_min_diff:',n_min_diff)
        print('10_max_diff:',n_max_diff)
        print('10_average_diff:',average_diff)
        print(sample_num_arr[num_index], ' sample test finished!')
        print('Time cost: ', average_time, 'seconds')
    plot_xy(x_arr=sample_num_arr, x_name='sample_num',
            y_arr=[n_min_diff_arr, n_max_diff_arr, n_average_diff_arr], y_legend_arr=['min', 'max', 'average'],
            y_name='color_diff_in_10',
            fig_name='random_test', fig_dir='fig_dir')


main()
