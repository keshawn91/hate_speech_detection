# -*- coding: utf-8 -*-
# src/__init__.py

# 可以导入包的主要功能，方便用户直接访问
from .data import *
from .model import *

# 或者显式定义__all__来控制from src import *的行为
__all__ = ['data', 'model', 'train', 'predict', 'evaluate']