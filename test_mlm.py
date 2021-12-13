import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import pipeline, BertTokenizer,BertTokenizer, BertForMaskedLM

import time
start = time.time()
# class Model(nn.Module):
#     def __init__(self, pretrain_model_path='trained_mlm_model/try_0', hidden_size=768):
#         super(Model, self).__init__()
#         self.pretrain_model_path = pretrain_model_path
#         self.bert = BertModel.from_pretrained(pretrain_model_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.dropout = nn.Dropout(0.1)
#         self.embed_size = hidden_size
#         self.cls = nn.Linear(self.embed_size, 2)

#     def forward(self, ids, attention_mask, labels=None, training=True):
#         loss_fct = nn.CrossEntropyLoss()

#         context = ids
#         types = None
#         mask = attention_mask
#         sequence_out, cls_out = self.bert(context, token_type_ids=types, attention_mask=attention_mask, output_all_encoded_layers=False)
#         cls_out = self.dropout(cls_out)
#         logits = self.cls(cls_out)

#         if training:
#           loss = loss_fct(logits.view(-1, 2), labels.view(-1))
          
#           return loss, nn.Softmax(dim=-1)(logits)
#         else:
#           return logits


# pretrain_model_path = 'bert_custom_tokenizer/vocab.txt'
# tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
# '''
# It can be modified by itself params
# '''

# CLS_TOKEN = '[CLS]'
# SEP_TOKEN = '[SEP]'
# seq_length = 64

# mybertmodel = Model()


vocab_file = 'bert_custom_tokenizer/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_file)




mlm_model = '/home/info/Aakash/bert custom/training_bert/checkpoint-520000'
x = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=mlm_model,return_dict=True)

end = time.time()
print("Load time", end - start)

# text = "म भात "  " अनि खेल्न जान्छु"
text = "आज को चुनाबमा एमालेले " + tokenizer.mask_token + " हात पार्यो।"

input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
output = x(**input)
logits = output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
for token in top_10:
   word = tokenizer.decode([token])
   new_sentence = text.replace(tokenizer.mask_token, word)
   print(new_sentence)
