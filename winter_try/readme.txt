限制配方使用的色浆种类在3种以下
1.尝试在损失函数中做约束，当色浆种类超过3种时，损失函数加大
2.尝试在最后预测配方时对配方进行过滤，只保留色浆种类数3种以下的配方


在没有进行约束时
test sample: 0
total_sample_num:  200000
color_diff<=10.0:  46461
color_diff<=1.0:  28
past_type<=3:  144871
diff<=10.0 and type<=3 : 37829
diff<=1.0 and type<=3 : 17