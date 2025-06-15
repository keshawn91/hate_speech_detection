# -*- coding: utf-8 -*-
import json
from typing import List, Dict
import re


def load_data(file_path: str) -> List[Dict]:
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_output(output: str) -> Dict:
    """解析output字段为结构化数据"""
    # 使用正则表达式匹配各部分
    match = re.match(r'(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*(\[END\])?', output)
    if not match:
        return {
            'target': 'NULL',
            'argument': 'NULL',
            'target_group': '其他',
            'hateful': 'non-hate'
        }

    return {
        'target': match.group(1).strip(),
        'argument': match.group(2).strip(),
        'target_group': match.group(3).strip(),
        'hateful': match.group(4).strip().lower()
    }


def preprocess_data(raw_data: List[Dict]) -> List[Dict]:
    """预处理原始数据"""
    processed = []
    for item in raw_data:
        # 训练数据有output字段
        if 'output' in item:
            parsed = parse_output(item['output'])
            processed.append({
                'text': item['content'],
                'target': parsed['target'],
                'argument': parsed['argument'],
                'target_group': parsed['target_group'],
                'hateful': parsed['hateful']
            })
        # 测试数据只有content
        else:
            processed.append({
                'text': item['content'],
                'target': 'NULL',
                'argument': 'NULL',
                'target_group': '其他',
                'hateful': 'non-hate'
            })
    return processed