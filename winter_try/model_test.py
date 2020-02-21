import numpy as np

np.set_printoptions(threshold=2 ** 20)
import torch
import copy
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from winter_try.info import base_color_num, reflectance_dim, base_color_names
from winter_try.utils import conc2ref_km, ciede2000_color_diff
from sklearn.manifold import TSNE
from time import time
from winter_try.k_means import kmeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)


# # 生成测试样本：在标准正态分布中采样，对配方和分光反射率数据做补充
def generate_test_sample(test_ref, N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total):
    test_samp = np.tile(np.array(test_ref), N_sample).reshape(N_sample, reflectance_dim)
    test_samp = torch.tensor(test_samp, dtype=torch.float)
    test_samp += y_noise_scale * torch.randn(N_sample, reflectance_dim)
    test_samp = torch.cat([torch.randn(N_sample, dim_z),  # zeros_noise_scale *
                           torch.zeros(N_sample, dim_total - dim_y - dim_z),
                           test_samp], dim=1)
    test_samp = test_samp.to(device)
    return test_samp


# 预测配方
def predict(concentrations, reflectance, test_samp, model):
    # 使用inn预测配方
    predict_formula = model(test_samp, rev=True)[:, :base_color_num]
    predict_formula = predict_formula.cpu().data.numpy()
    # 假设涂料浓度小于一定值时，就不需要这种涂料
    predict_formula = np.where(predict_formula < 0.1, 0, predict_formula)
    return predict_formula


# 用于判断配方是否和已有的配方使用了相同的颜色种类
def formula_like_exist(formula, top_n, n):
    for i in range(n):
        if (formula in top_n[i]):
            return i
    return n - 1


# 根据分光反射率预测出配方，对于每一个分光反射率，选择色差最小的十个配方
# concentrations:目标真实配方，reflectance:目标分光反射率，predict_formula:预测的配方
def test1(concentrations, reflectance, predict_formula):
    t_start = time()
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top10 = [[100, 0], [100, 0], [100, 0], [100, 0], [100, 0],
             [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        if diff < top10[9][0]:
            top10[9][0] = diff
            top10[9][1] = n
            top10.sort()
    # 打印预测结果
    print('real_formula:')
    print(concentrations)
    for n in range(10):
        print('predict_formula_', n)
        print(predict_formula[top10[n][1], :])
        print("color diff: %.2f \n" % top10[n][0])
    # 将预测结果输出到excel
    workbook = xlsxwriter.Workbook("excel_dir/top10.xlsx")
    worksheet = workbook.add_worksheet('top10')
    # 填写配方编号
    for n in range(base_color_num):
        worksheet.write(0, n + 1, base_color_names[n])
    # 填写真实配方
    worksheet.write(1, 0, 'Real Formula')
    for n in range(base_color_num):
        worksheet.write(1, n + 1, concentrations[n])
    # 填写预测配方
    for i in range(10):
        worksheet.write(i + 2, 0, 'Predict formula %s' % i)
        temp_formula = predict_formula[top10[i][1], :]
        for n in range(base_color_num):
            worksheet.write(i + 2, n + 1, temp_formula[n])
        worksheet.write(i + 2, base_color_num + 1, top10[i][0])
    worksheet.write(i + 3, 0, 'time_cost')
    worksheet.write(i + 3, 1, time() - t_start)
    workbook.close()
    print('time_cost: ', time() - t_start)
    print("\n\n")


# 选择满足色浆种类不超过3种，且色差最小的三个配方
# concentrations:目标真实配方，reflectance:目标分光反射率，predict_formula:预测的配方
def test2(concentrations, reflectance, predict_formula):
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top10 = [[100, -1], [100, -1], [100, -1], [100, -1], [100, -1], [100, -1], [100, -1], [100, -1], [100, -1],
             [100, -1]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        # 只选择使用的色浆种类小于等于3种的配方
        if (sum(temp_formula) <= 3 and diff < top10[9][0]):
            top10[9][0] = diff
            top10[9][1] = n
            top10.sort()
    print('real_formula:')
    print(concentrations)
    for n in range(10):
        if (top10[n][0] == 100 or top10[n][1] == -1):
            print('No more qualified formula!')
            break
        else:
            print('predict_formula_', n)
            print(predict_formula[top10[n][1], :])
            print("color diff: %.2f \n" % top10[n][0])
    print("\n\n")


# 选择满足色浆种类不超过3种，且色差最小的三个配方,且如果配方的成分相同，只取最小的那一个
# concentrations:目标真实配方，reflectance:目标分光反射率，predict_formula:预测的配方
def test3(concentrations, reflectance, predict_formula):
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top10 = [[100, -1, []], [100, -1, []], [100, -1, []], [100, -1, []], [100, -1, []],
             [100, -1, []], [100, -1, []], [100, -1, []], [100, -1, []], [100, -1, []]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        # 只选择使用的色浆种类小于等于3种的配方
        if (sum(temp_formula) <= 3 and diff < top10[9][0] and diff <= 1.0):
            temp_formula = list(temp_formula)
            index = formula_like_exist(temp_formula, top10, 10)
            if top10[index][0] > diff:
                top10[index][0] = diff
                top10[index][1] = n
                top10[index][2] = temp_formula
            top10.sort()
    print('real_formula:')
    print(concentrations)
    for n in range(10):
        if (top10[n][0] == 100 or top10[n][1] == -1):
            print('No more qualified formula!')
            break
        else:
            print('predict_formula_', n)
            print(predict_formula[top10[n][1], :])
            print("color diff: %.2f \n" % top10[n][0])
    print("\n\n")


# 统计预测的配方结果
def sample_count(concentrations, reflectance, predict_formula):
    # 选取的样本总数
    print('total_sample_num: ', predict_formula.shape[0])
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    color_diff_under_1_num = 0
    color_diff_under_10_num = 0
    paste_type_under_3_num = 0
    num_1_3 = 0
    num_10_3 = 0
    for n in range(predict_formula.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        temp_count = sum(temp_formula)
        # 只选择使用的色浆种类小于等于3种的配方
        if temp_count <= 3:
            paste_type_under_3_num += 1
        if diff <= 10.0:
            color_diff_under_10_num += 1
            if diff <= 1.0:
                color_diff_under_1_num += 1
        if temp_count <= 3 and diff <= 10.0:
            num_10_3 += 1
            if diff <= 1.0:
                num_1_3 += 1
    print('color_diff<=10.0: ', color_diff_under_10_num)
    print('color_diff<=1.0: ', color_diff_under_1_num)
    print('past_type<=3: ', paste_type_under_3_num)
    print('diff<=10.0 and type<=3 :', num_10_3)
    print('diff<=1.0 and type<=3 :', num_1_3)


# 统计配方的种类以及每种配方出现的次数
def sample_count2(concentrations, reflectance, predict_formula):
    # 用一个字典来保存每种配方出现的次数
    formula_dict = {}
    # 计算分光发射率
    formula_ref = conc2ref_km(predict_formula)
    for n in range(predict_formula.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        if diff > 5:
            continue
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        dict_key = tuple(temp_formula)
        if dict_key in formula_dict:
            formula_dict[dict_key] += 1
        else:
            formula_dict[dict_key] = 1
    # 将预测结果输出到excel
    workbook = xlsxwriter.Workbook("excel_dir/fcount2.xlsx")
    worksheet = workbook.add_worksheet('count')
    # 填写配方编号
    for n in range(base_color_num):
        worksheet.write(0, n, base_color_names[n])
    # 填写配方种数
    index = 1
    for sample in formula_dict:
        for n in range(base_color_num):
            worksheet.write(index, n, sample[n])
        worksheet.write(index, base_color_num, formula_dict[sample])
        index += 1
    workbook.close()
    return formula_dict


# 对色差小于2的配方种类，个数最多的前3种分别计算聚类中心
def k_means_seperate2(concentrations, reflectance, predict_formula):
    formula_dict = sample_count2(concentrations, reflectance, predict_formula)
    formula_dict = dict(sorted(formula_dict.items()), key=lambda item: item[1])
    # 保存数量前10的配方种类
    count = 0
    top3_dict = {}
    for key, value in formula_dict.items():
        if count >= 3:
            break
        top3_dict[key] = count
        count += 1
    # 筛选种类数量前10的配方保存
    list = [[], [], []]
    for n in range(predict_formula.shape[0]):
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        dict_key = tuple(temp_formula)
        if dict_key in top3_dict:
            list[top3_dict[dict_key]].append(predict_formula[n, :])
    for f_list in list:
        centroids, cluster = kmeans(f_list, 1)
        formula_ref = conc2ref_km(np.array(centroids))
        diff = ciede2000_color_diff(reflectance, formula_ref[0, :])
        print(centroids[0])
        print(diff)


# 对色差小于10的配方种类，个数最多的前10种分别计算重心
def k_means_seperate(concentrations, reflectance, predict_formula):
    formula_dict = sample_count2(concentrations, reflectance, predict_formula)
    formula_dict = dict(sorted(formula_dict.items()), key=lambda item: item[1])
    # 保存数量前10的配方种类
    count = 0
    top10_dict = {}
    for key, value in formula_dict.items():
        if count >= 10:
            break
        top10_dict[key] = count
        count += 1
    # 筛选种类数量前10的配方保存
    list = [[], [], [], [], [], [], [], [], [], []]
    for n in range(predict_formula.shape[0]):
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        dict_key = tuple(temp_formula)
        if dict_key in top10_dict:
            list[top10_dict[dict_key]].append(predict_formula[n, :])
    # 对种类前10的配方分别求聚类中心及色差,聚类中心尝试1-10
    for c in range(1, 6, 1):
        print('centroids_num:%s' % c)
        # 聚类中心确定后，对于每一种配方求聚类中心并计算色差
        f_count = 0  # 记录是第几个配方
        for f_list in list:
            print('example:%s' % f_count)
            centroids, cluster = kmeans(f_list, c)
            formula_ref = conc2ref_km(np.array(centroids))
            for i in range(c):
                diff = ciede2000_color_diff(reflectance, formula_ref[i, :])
                print(centroids[i])
                print('color_diff:%s'%diff)
            f_count += 1
        print('-----------------------------------')


# 对所有预测配方寻找k-means中心计算色差
def k_means_center(concentrations, reflectance, predict_formula):
    predict_formula = list(predict_formula)
    centroids, cluster = kmeans(predict_formula, 10)
    formula_ref = conc2ref_km(np.array(centroids))
    for n in range(centroids.__len__()):
        print('predict formula: ', centroids[n])
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        print('diff:', diff)
    # 将结果输出到excel
    workbook = xlsxwriter.Workbook("excel_dir/kmeans1.xlsx")
    worksheet = workbook.add_worksheet('kmeans')
    # 填写配方编号
    for n in range(base_color_num):
        worksheet.write(0, n, base_color_names[n])
    for n in range(centroids.__len__()):
        for i in range(base_color_num):
            worksheet.write(n + 1, i, centroids[n][i])
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        worksheet.write(n + 1, base_color_num, diff)
    workbook.close()


# 将生成的配方降维可视化
def tsne_plot(concentrations, reflectance, predict_formula):
    target = []  # 每种配方应该是什么颜色
    next_color = 0  # 下一种画图颜色的编号
    color_dict = {}  # 用于保存配方和颜色的对应关系
    for n in range(predict_formula.shape[0]):
        temp_formula = copy.copy(predict_formula[n, :])
        temp_formula[temp_formula != 0] = 1
        dict_key = tuple(temp_formula)
        if dict_key in color_dict:
            target.append(color_dict[dict_key])
        else:
            color_dict[dict_key] = next_color
            target.append(next_color)
            next_color += 1
    # 二维可视化
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # low_dim_embs = tsne.fit_transform(predict_formula)
    # plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], s=1,c=target)
    # plt.savefig('tsne20w.png')
    # 三维可视化
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(predict_formula)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], low_dim_embs[:, 2], c=target, s=1)
    plt.savefig('tsne3D5k')


def main():
    # 加载模型
    inn = torch.load('model_dir/model_01')
    # 读取数据集
    data = np.load('data_dir/data_01.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    # 加载数据
    testsplit = 1
    test_conc = concentrations[:testsplit]
    test_ref = reflectance[:testsplit]
    # 选取的样本数
    N_sample = 200000
    y_noise_scale = 3e-2
    dim_x = base_color_num
    dim_y = reflectance_dim
    dim_z = 13
    dim_total = max(dim_x, dim_y + dim_z)
    # 进行测试
    for i in range(testsplit):
        test_samp = generate_test_sample(test_ref[i], N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total)
        print('test sample:', i)
        predict_formula = predict(test_conc[i], test_ref[i], test_samp, inn)
        # test1(test_conc[i], test_ref[i], predict_formula)
        # test2(test_conc[i], test_ref[i], predict_formula)
        # sample_count(test_conc[i], test_ref[i], predict_formula)
        # sample_count2(test_conc[i], test_ref[i], predict_formula)
        # tsne_plot(test_conc[i], test_ref[i], predict_formula)
        # k_means_center(test_conc[i], test_ref[i], predict_formula)
        # k_means_seperate(test_conc[i], test_ref[i], predict_formula)
        k_means_seperate2(test_conc[i], test_ref[i], predict_formula)


main()
