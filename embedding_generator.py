import torch

from transformers import BertTokenizer, BertForMaskedLM
from scipy.spatial.distance import cosine 

import os 


vocab_file = 'bert_custom_tokenizer/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_file)




mlm_model = '/home/info/Aakash/bert custom/training_bert/checkpoint-530000'
x = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=mlm_model,output_hidden_states=True)


def get_word_embedding_bert(input_text,tokenizer,model):
    marked_text = " [CLS] " + input_text + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)
    for i, token_str in enumerate(tokenized_text):
      print (i, token_str)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) 
    
    # Convert inputs to Pytorch tensors
    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = model(tokens_tensors, segments_tensors)
        # removing the first hidden state
        # the first state is the input state 
        hidden_states = outputs.hidden_states
    
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)


    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

def get_bert_embedding_sentence(input_sentence,tokenizer,model):

    marked_text = " [CLS] " + input_sentence + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)
    for i, token_str in enumerate(tokenized_text):
      print (i, token_str)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) 
    
    # Convert inputs to Pytorch tensors
    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = model(tokens_tensors, segments_tensors)
        # removing the first hidden state
        # the first state is the input state 

        hidden_states = outputs.hidden_states
        # second_hidden_states = outputs[2]
    # `hidden_states` has shape [13 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # import pdb;pdb.set_trace()
    return sentence_embedding


if __name__== "__main__":
    import time
    start = time.time()
    test_sentence = 'सयर बजारमा नेप्से भरि अंकले गिरावट'
    test_sentence_2 = "कांग्रेसले चुनाबमा एमालेसंग भारि मातले हार भेहोर्यो "
    test_sentence_3 = "कांग्रेसले एमाले संग वार्ता बसेर सम्विधानको धारा २ को निस्कर्ष निकाल्यो "
    embedding1 = get_word_embedding_bert(test_sentence,tokenizer,x)
    embedding2 = get_word_embedding_bert(test_sentence_2,tokenizer,x)
    embedding3 = get_word_embedding_bert(test_sentence_3,tokenizer,x)
    """
    #For word embedding
    embedding1 = get_word_embedding_bert(test_sentence,tokenizer,x)
    embedding2 = get_word_embedding_bert(test_sentence_2,tokenizer,x)
    embedding3 = get_word_embedding_bert(test_sentence_3,tokenizer,x)
    embedding_1_haar = embedding1[6]
    embedding_2_haar = embedding2[13]
    embedding_3_haar = embedding3[7]
    cos_dist_different = 1 - cosine(embedding_1_haar,embedding_2_haar)
    cos_dist_same = 1 - cosine(embedding_2_haar,embedding_3_haar)
    print("Difference",cos_dist_different)
    print("Similar",cos_dist_same)
    end = time.time()
    print("Time",end-start)
    """
    #For sentence embedding
    embedding1 = get_bert_embedding_sentence(test_sentence,tokenizer,x)
    embedding2 = get_bert_embedding_sentence(test_sentence_2,tokenizer,x)
    embedding3 = get_bert_embedding_sentence(test_sentence_3,tokenizer,x)
    cos_dist_different = 1 - cosine(embedding1,embedding2)
    cos_dist_same = 1 - cosine(embedding2,embedding3)
    print("Difference",cos_dist_different)
    print("Similar",cos_dist_same)
    end = time.time()
    print("Time",end-start)