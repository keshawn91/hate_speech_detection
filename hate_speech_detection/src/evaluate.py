# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score
import difflib


def calculate_hard_f1(preds, golds):
    """计算硬匹配F1"""
    return f1_score(golds, preds, average='macro')


def calculate_soft_f1(preds, golds):
    """计算软匹配F1"""
    # 实现软匹配逻辑
    pass


def evaluate():
    # 加载预测结果和标准答案
    with open('../outputs/predictions.txt') as f:
        preds = [line.strip() for line in f]

    with open('../data/raw/test1.json') as f:
        golds = json.load(f)

    # 计算指标
    hard_f1 = calculate_hard_f1(preds, golds)
    soft_f1 = calculate_soft_f1(preds, golds)
    avg_f1 = (hard_f1 + soft_f1) / 2

    print(f'Hard F1: {hard_f1:.4f}')
    print(f'Soft F1: {soft_f1:.4f}')
    print(f'Average F1: {avg_f1:.4f}')


if __name__ == '__main__':
    evaluate()