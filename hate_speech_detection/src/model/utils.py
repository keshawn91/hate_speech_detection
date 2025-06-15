# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer
from typing import Dict, List, Tuple, Optional


# ====================== 模型训练辅助工具 ======================

def compute_metrics(
        pred_logits: torch.Tensor,
        gold_labels: torch.Tensor,
        average: str = 'macro'
) -> Dict[str, float]:
    """计算分类任务的评估指标（F1, Precision, Recall）

    Args:
        pred_logits: 模型输出的 logits (batch_size, num_classes)
        gold_labels: 真实标签 (batch_size,)
        average: F1计算方式 ('micro', 'macro', 'weighted')

    Returns:
        Dict[str, float]: 包含 precision, recall, f1 的字典
    """
    preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
    golds = gold_labels.cpu().numpy()

    report = classification_report(
        golds, preds,
        target_names=["non-hate", "hate"],
        output_dict=True,
        zero_division=0
    )

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }


def compute_multi_task_metrics(
        target_group_preds: torch.Tensor,
        target_group_golds: torch.Tensor,
        hate_preds: torch.Tensor,
        hate_golds: torch.Tensor,
) -> Dict[str, float]:
    """计算多任务（目标群体分类 + 仇恨言论分类）的评估指标"""
    target_group_metrics = compute_metrics(target_group_preds, target_group_golds)
    hate_metrics = compute_metrics(hate_preds, hate_golds)

    return {
        "target_group_f1": target_group_metrics["f1"],
        "hate_f1": hate_metrics["f1"],
        "avg_f1": (target_group_metrics["f1"] + hate_metrics["f1"]) / 2,
    }


# ====================== 数据处理辅助工具 ======================

def tokenize_and_align_labels(
        tokenizer: BertTokenizer,
        texts: List[str],
        max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """使用 tokenizer 对文本进行编码，并返回 PyTorch 张量

    Args:
        tokenizer: BertTokenizer
        texts: 待编码的文本列表
        max_length: 最大长度

    Returns:
        Dict[str, torch.Tensor]: 包含 input_ids, attention_mask 的字典
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


# ====================== 模型保存与加载 ======================

def save_model(
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
) -> None:
    """保存模型、优化器和学习率调度器

    Args:
        model: 要保存的模型
        path: 保存路径（如 `models/bert_hate_speech.bin`）
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    """
    state = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)


def load_model(
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """加载模型、优化器和学习率调度器

    Args:
        model: 要加载的模型
        path: 模型路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 加载设备（'cuda' 或 'cpu'）
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return model


# ====================== 其他工具函数 ======================

def set_seed(seed: int = 42) -> None:
    """设置随机种子（确保实验可复现）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """返回可用设备（'cuda' 或 'cpu'）"""
    return "cuda" if torch.cuda.is_available() else "cpu"