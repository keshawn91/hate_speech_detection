# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # 定义标签映射
        self.target_group_map = {
            'Racism': 0, 'Gender': 1, 'LGBTQ': 2, 'others': 3, 'other': 3, '其他': 3, 'non-hate': 4
        }
        self.hate_map = {'hate': 1, 'non-hate': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 训练数据有标签
        if self.is_train:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'target_group': torch.tensor(
                    self.target_group_map.get(item['target_group'], 3),  # 默认为"其他"
                    dtype=torch.long
                ),
                'hateful': torch.tensor(
                    self.hate_map.get(item['hateful'], 0),  # 默认为non-hate
                    dtype=torch.long
                )
            }
        # 测试数据只有文本
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': item['text']  # 保留原始文本用于预测输出
            }