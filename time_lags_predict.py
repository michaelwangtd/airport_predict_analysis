#!/usr/bin/env python
# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import inout
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

"""
    参考 http://www.leiphone.com/news/201703/6rVkgxvxUumnv5mm.html文章

    提取了滞后时间序列变量，进行预测

    问题：
        1）类似这样的文本数据太多，对每一个文本数据执行下面的过程，回归决策树的模型参数不同，效果不好
"""

if __name__ == '__main__':
    pd.set_option('display.width', 300)
    np.set_printoptions(linewidth=300, suppress=True)

    # 这里的滞后特征列表可以由之前的过程得到
    lags_feature_list = ['t-6', 't-3', 't-2', 't-1', 't']
    forecast_len = 11

    inFileName = 'lags_features_clean.csv'
    # inFileName = 'lags_features_clean_O1002.csv'
    # inFilePath = inout.getDataModelPipelinePath(inFileName)
    inFilePath = inout.getDataPath(inFileName)
    data = pd.read_csv(inFilePath, header=0)
    data = data[lags_feature_list].values
    # print data
    # 先把训练集/测试集数据分开
    train_data = data[:70]
    test_data = data[70:]
    # print type(test_data[:,-1])
    # exit(0)
    # print len(train_data)
    # print len(test_data)
    # 初始化预测查询表
    x_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    # print x_train
    # print type(x_train)
    # print x_train.shape()
    # print y_train
    # print type(y_train)
    # exit(0)

    base_num = len(y_train)
    predict_search_list = list(y_train)

    alpha_can = np.logspace(-3, 2, 10)

    # model = Ridge()
    # lr = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lr = DecisionTreeRegressor(criterion='mse', max_depth=20)
    # lr = LogisticRegression()
    # lr = LinearRegression()
    # lr = svm.SVR(kernel='rbf',gamma=0.2,C=100)
    # x_train = x_train.reshape(-1,1)
    lr.fit(x_train, y_train)

    y_hat_list = []
    # 滚动预测
    for i in range(0, forecast_len):
        curr_index = i + base_num
        predict_queue = []
        predict_queue.append(predict_search_list[curr_index - 6])
        predict_queue.append(predict_search_list[curr_index - 3])
        predict_queue.append(predict_search_list[curr_index - 2])
        predict_queue.append(predict_search_list[curr_index - 1])
        predict_queue = np.array(predict_queue)
        print i, predict_queue
        print type(predict_queue)
        # exit(0)

        y_hat = lr.predict(predict_queue)
        print 'result:', y_hat
        y_hat_list.append(y_hat[0])
        predict_search_list.append(y_hat[0])
    print len(y_hat_list), y_hat_list
    print len(test_data[:, -1]), test_data[:, -1]
    # exit(0)
    plt.figure()
    ticks = [i for i in range(len(y_hat_list))]
    y_hat = np.array(y_hat_list)
    plt.subplot(1, 1, 1)
    plt.plot(ticks, y_hat, '-b')
    plt.plot(ticks, test_data[:, -1], '-r')
    plt.show()
    # pyplot.bar(ticks,fit.ranking_)
    # pyplot.xticks(ticks,names)