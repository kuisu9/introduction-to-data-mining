# -*- code:utf-8 -*-

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # 导入sklearn的贝叶斯算法
from sklearn import metrics


# 1.按类别划分样本
# separated = {0: [[att1, att2, ... att8, 0], ...],
#             1: [[att1, att2, ... att8, 1], [att1, att2, ... att8, 1], ...]}
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# 2.提取属性特征. 对一个类的所有样本,计算每个属性的均值和方差,
# summaries = [(att1_mean,att1_stdev), (att2_mean,att2_stdev), .., (att8_mean,att8_stdev)]
def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# 3.按类别提取属性特征
# summaries = {0:[(att1_mean,att1_stdev), (att2_mean,att2_stdev), .., (att8_mean,att8_stdev)],
#             1:[(att1_mean,att1_stdev), (att2_mean,att2_stdev), .., (att8_mean,att8_stdev)]}
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    keyList = list(separated.keys())
    for classValue in keyList:
        summaries[classValue] = summarize(separated[classValue])
    return summaries


# 计算高斯概率密度函数. 计算样本的某一属性x的概率,归属于某个类的似然
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# 4.对一个样本,计算它属于每个类的概率
def calculate_class_probabilities(summaries, inputVector):
    probabilities = {}
    keyList = list(summaries.keys())
    for classValue in keyList:
        probabilities[classValue] = 1
        for i in range(len(summaries[classValue])):  # 属性个数
            mean, stdev = summaries[classValue][i]  # 训练得到的第i个属性的提取特征
            x = inputVector[i]  # 测试样本的第i个属性x
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities


# 5.单个数据样本的预测. 找到最大的概率值,返回关联的类
def predict(summaries, inputVector):
    probabilities = calculate_class_probabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    keyList = list(probabilities.keys())
    for classValue in keyList:
        if bestLabel is None or probabilities[classValue] > bestProb:
            bestProb = probabilities[classValue]
            bestLabel = classValue
    return bestLabel


# 多个数据样本的预测
def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# 6.计算精度
def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet)))


# 7.计算精确率
def get_precision(testSet, predictions):
    truePositive = 0
    positive = 0
    for x in range(len(testSet)):
        if predictions[x] == 1:
            positive += 1
            if testSet[x][-1] == predictions[x]:
                truePositive += 1
    return (truePositive / positive)


# 8.计算召回率
def get_recall(testSet, predictions):
    truePositive = 0
    positive = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == 1:
            positive += 1
            if testSet[x][-1] == predictions[x]:
                truePositive += 1
    return (truePositive / positive)


def main():
    # 读取数据
    filename = 'pima-indians-diabetes.csv'
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.sample(frac=1.0)
    y = dataset[8]
    X = dataset[[0, 1, 2, 3, 4, 5, 6, 7]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 2, random_state=1, stratify=y)
    dataset = np.array(dataset)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train = np.array(train)
    test = np.array(test)

    # 随机划分数据:1/2训练和1/3测试
    trainSize = int(len(train))
    randomIdx = [i for i in range(len(train))]
    trainSet = []
    testSet = []
    trainSet.extend(train[idx, :] for idx in randomIdx[:trainSize])
    testSet.extend(test[idx, :] for idx in randomIdx[:trainSize])

    # 计算模型
    summaries = summarize_by_class(trainSet)

    # 用测试数据集测试模型
    predictions = get_predictions(summaries, testSet)
    accuracy = get_accuracy(testSet, predictions)
    precision = get_precision(testSet, predictions)
    recall = get_recall(testSet, predictions)
    f = 2 * precision * recall / (precision + recall)
    print(('Accuracy:{0}').format(accuracy))
    print(('Precision:{0}').format(precision))
    print(('Recall:{0}').format(recall))
    print(('F1:{0}').format(f))

    # Scikit - Learn中朴素贝叶斯算法分类器
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_pred = model.predict(X_test)
    print('Accuracy：', metrics.accuracy_score(y_test, y_pred))
    print('Precision：', metrics.precision_score(y_test, y_pred))
    print('Recall：', metrics.recall_score(y_test, y_pred))
    print('F1：', metrics.f1_score(y_test, y_pred))


if __name__ == '__main__':
    main()