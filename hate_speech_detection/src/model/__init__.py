# -*- coding: utf-8 -*-
# src/model/__init__.py

# 暴露model模块的主要接口
from .model import HateSpeechModel
from .utils import *  # 如果有utils.py的话

__all__ = ['HateSpeechModel']