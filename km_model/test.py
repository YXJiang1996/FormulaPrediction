import numpy as np
import torch
from km_model.info import base_color_num
from km_model.utils import conc2ref_km, ciede2000_color_diff


def test(concentrations, reflectance, model):
    # 使用inn预测配方
    predict_formula = model(reflectance, rev=True)[:, :base_color_num]
    predict_formula = predict_formula.cpu().data.numpy()
    # 假设涂料浓度小于一定值时，就不需要这种涂料
    predict_formula = np.where(predict_formula < 0.1, 0, predict_formula)

    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top3 = [[100, 0], [100, 0], [100, 0]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        if diff < top3[2][0]:
            top3[2][0] = diff
            top3[2][1] = n
            top3.sort()
    print('real_formula:', concentrations)
    for n in range(3):
        print('predict_formula_', n, formula_ref[top3[n][1], :])
        print("color diff: %.2f \n" % top3[n][0])
    print("\n\n")


def main():
    # 加载模型
    inn = torch.load('km_model/models/model_01')
    # 读取数据集
    data = np.load('km_model/dataset/3in21.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    # 加载数据
    testsplit = 3 * 3
    test_conc = concentrations[:testsplit]
    test_ref = reflectance[:testsplit]
    # 进行测试
    for i in range(testsplit):
        # test(test_conc[i],test_ref[i],inn)
        print(test_conc[i])
        print(test_ref[i])
main()