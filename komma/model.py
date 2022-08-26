import torch.nn as nn

from transformers import BertConfig, BertModel, BertTokenizer

class CommaBERT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(CommaBERT, self).__init__()

        segment_size = config["segment_size"]
        bert_uri = config["bert_uri"]

        bert_config = BertConfig.from_pretrained(bert_uri, output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained(bert_uri)
        self.bert = BertModel.from_pretrained(bert_uri, config=bert_config)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.batch_norm = nn.BatchNorm1d(segment_size * bert_config.vocab_size)
        self.linear = nn.Linear(segment_size * bert_config.vocab_size)
        self.dropout = nn.Dropout(config["dropout"])
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.bert(x)
        x = x.view(x.shape[0], -1)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return self.activation(x)
