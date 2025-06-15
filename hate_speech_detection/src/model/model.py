# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class HateSpeechModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.target_classifier = nn.Linear(config.hidden_size, 2)  # BIO标注
        self.argument_classifier = nn.Linear(config.hidden_size, 2)  # BIO标注
        self.target_group_classifier = nn.Linear(config.hidden_size, 5)  # 5类目标群体
        self.hate_classifier = nn.Linear(config.hidden_size, 2)  # hate/non-hate

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        target_logits = self.target_classifier(sequence_output)
        argument_logits = self.argument_classifier(sequence_output)
        target_group_logits = self.target_group_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)

        return {
            'target_logits': target_logits,
            'argument_logits': argument_logits,
            'target_group_logits': target_group_logits,
            'hate_logits': hate_logits
        }