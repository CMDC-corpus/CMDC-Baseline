import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import math


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    print(len(test_preds))
    print(len(test_truth))
    predString = ''

    index = 0
    length = len(test_truth)
    mae = 0
    rmse = 0
    while (index < length):
        mae += abs(test_preds[index] - test_truth[index])
        rmse += (test_preds[index] - test_truth[index]) * (test_preds[index] - test_truth[index])
        index += 1
    rmse = math.sqrt(rmse / length)
    mae = mae / length
    print('mae = ' + str(mae))
    print('rmse = ' + str(rmse))
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])



def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


##计算auroc的公式
def evaluate_auroc(results, truths):
    resultClassify = []
    truthClassify = []
    for value in results:
        if (value < 10):
            resultClassify.append(0)
        else:
            resultClassify.append(1)
    for value in truths:
        if (value < 10):
            truthClassify.append(0)
        else:
            truthClassify.append(1)
    truthClassify = np.array(truthClassify)
    resultClassify = np.array(resultClassify)
    auroc = roc_auc_score(truthClassify, resultClassify)
    return auroc


##计算CCC系数的公式
def evaluate_ccc(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    covariance = np.nanmean((x - x_mean) * (y - y_mean))
    x_var = np.nanmean((x - x_mean) ** 2)
    y_var = np.nanmean((y - y_mean) ** 2)
    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)
    return CCC


##计算pearson系数
def pearson(x, y):
    pccs = pearsonr(x, y)
    return pccs[0]


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        print(test_preds)
        print(test_preds.shape)
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        print(test_truth)
        print(test_truth.shape)
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            array = test_preds[:, emo_ind]
            print(array)
            print(array.shape)
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            print(test_preds_i.shape)
            print(test_preds_i)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


##results是最终的结果
##truths是实际的标签变量
def eval_yiyu(results, truths):
    ##test_preds:[6,1,2]
    test_preds = results.view(-1, 1, 2).cpu().detach().numpy()
    ##test_truth:[6,1]
    test_truth = truths.view(-1, 1).cpu().detach().numpy()
    ##取0位置的
    test_preds_i = np.argmax(test_preds[:, 0], axis=1)
    test_truth_i = test_truth[:, 0]
    ##将信息写入到文件中
    trueInfo = ''
    predInfo = ''
    index = 0
    length = len(test_truth)
    ccc = evaluate_ccc(test_preds, test_truth)
    auroc = evaluate_auroc(test_preds, test_truth)
    pp = pearson(test_preds, test_truth)
    print('ccc = ' + str(ccc))
    print('auroc = ' + str(auroc))
    print('pp = ' + str(pp))
    zeroZero = 0
    zeroOne = 0
    oneZero = 0
    oneOne = 0
    index = 0
    while (index < length):
        if (str(test_truth_i[index]) == '0' and str(test_preds_i[index]) == '0'):
            zeroZero += 1
        if (str(test_truth_i[index]) == '0' and str(test_preds_i[index]) == '1'):
            zeroOne += 1
        if (str(test_truth_i[index]) == '1' and str(test_preds_i[index]) == '0'):
            oneZero += 1
        if (str(test_truth_i[index]) == '0' and str(test_preds_i[index]) == '1'):
            oneOne += 1
        index += 1
    precision = zeroZero / (zeroZero + zeroOne)
    recall = zeroZero / (zeroZero + oneZero)
    f1 = 2 * precision * recall / (precision + recall)
    print('precision = ' + str(precision))
    print('recall = ' + str(recall))
    print('f1 = ' + str(f1))



