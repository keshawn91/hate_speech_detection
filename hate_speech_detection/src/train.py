# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from src.data.dataset import HateSpeechDataset
from src.data.preprocess import load_data, preprocess_data
from src.model.model import HateSpeechModel
from src.model.utils import compute_metrics, save_model, set_seed, get_device

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))


def train():
    # 加载配置
    with open('configs/default_config.json') as f:
        config = json.load(f)

    set_seed(42)
    device = get_device()

    # 加载并预处理数据
    raw_data = load_data('data/raw/train.json')
    processed_data = preprocess_data(raw_data)

    # 初始化模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    model = HateSpeechModel.from_pretrained(config['model_name'])
    model.to(device)

    # 创建数据集和数据加载器
    dataset = HateSpeechDataset(processed_data, tokenizer, config['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # 准备输入
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_groups = batch['target_group'].to(device)
            hate_labels = batch['hateful'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算多任务损失
            loss1 = criterion(outputs['target_group_logits'], target_groups)
            loss2 = criterion(outputs['hate_logits'], hate_labels)
            loss = loss1 + loss2

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 每个epoch结束后打印指标
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {avg_loss:.4f}')

    # 保存模型
    save_model(model, 'models/model.pth', optimizer)


if __name__ == '__main__':
    train()