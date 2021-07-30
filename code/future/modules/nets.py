from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

import torch.nn as nn
import torch


class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self):
        raise NotImplementedError


class BertForSequenceTagging(LinearPredictor):
    """
    used for both tagging and ner.
    """

    def __init__(self, bert_config, out_dim, dropout=0.1):
        super(BertForSequenceTagging, self).__init__(bert_config, out_dim, dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)
        logits = bert_out[if_tgts]
        return (
            logits,
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )
