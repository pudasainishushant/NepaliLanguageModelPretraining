from torch.multiprocessing import TimeoutError, Pool,set_start_method,Queue
import torch.multiprocessing as mp
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import json
import math

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def load_model():
    model = BertForMaskedLM.from_pretrained('training_bert/checkpoint-140000').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert_custom_tokenizer/vocab.txt')
    return tokenizer, model

tokenizer, model =load_model()
#st.text('Done!')
import pdb;pdb.set_trace()
def score(sentence):
    if len(sentence.strip().split())<=1 : return 10000
    tokenize_input = tokenizer.tokenize(sentence)
    if len(tokenize_input)>512: return 10000
    input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0).to(device)
    with torch.no_grad():
        loss=model(input_ids)[0]
    print(loss)
    return  math.exp(loss.item()/len(tokenize_input))

sentence = "प्रधानमन्त्री आज काठमाडौं "
print(score(sentence))


