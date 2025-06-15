# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from src.model.model import HateSpeechModel
from src.data.dataset import HateSpeechDataset
import json


def predict():
    # 加载配置
    with open('../configs/default_config.json') as f:
        config = json.load(f)

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    model = HateSpeechModel.from_pretrained(config['model_name'])
    model.load_state_dict(torch.load('../models/model.pth'))

    # 加载测试数据
    test_data = load_data('../data/raw/default_config.json')

    # 预测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    results = []
    for item in test_data:
        # 实现预测逻辑
        pass

    # 保存结果
    with open('../outputs/predictions.txt', 'w') as f:
        f.write('\n'.join(results))


if __name__ == '__main__':
    predict()