from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import torch
from glob import glob
from tqdm import tqdm

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


# path of the tokenizer model
vocab_file = 'bert_custom_tokenizer/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file)

vocab_size = 30522
total_epochs = 5

# path of the dataset from where training will be done
# dataset_file_path = 'dataset/0.txt'
# dataset_file_path = glob('dataset/*.txt')
eval_dataset_file_path = "dataset/0.txt"

# all_text = ""

# for f in tqdm(dataset_file_path):
#     with open(f, 'r') as file:
#         # print(len(file.read()))
#         all_text = " ".join([all_text, file.read()])
#     # print(len(all_text))

# with open("final_text.txt",'w') as file:
#     file.write(all_text)

dataset_file_path = "final_text.txt"

# path to save the trained mlm model
mlm_model_path = 'trained_mlm_model/try_1/'

#folder to store the training arguments results
output_dir = 'training_bert/'

config = BertConfig(
    vocab_size = vocab_size,
    hidden_size = 768,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    max_position_embeddings = 512
)



# model = BertForMaskedLM(config).from_pretrained('bert_model')
model = BertForMaskedLM(config)
print(f"Modle Type: - {type(model)}")
model.resize_token_embeddings(len(tokenizer))
print('No of parameters: ',model.num_parameters())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = dataset_file_path,
    block_size = 64  # maximum sequence length
)


eval_dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = eval_dataset_file_path,
    block_size = 64  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=total_epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_steps=10000,
    do_eval=True,
    do_train = True,
    evaluation_strategy = "steps", 
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)


print(f"Trainer Type: - {type(trainer)}")
# trainer.to(device)


trainer.train()
trainer.save_model(mlm_model_path)

print(trainer.state.log_history)



## Evaluating perplexity of the masked language model
import math
eval_results = trainer.evaluate()
print(f"Perplexity : {math.exp(eval_results['eval_loss']):.2f}")


# # Keep track of train and evaluate loss.
# loss_history = {'train_loss':[], 'eval_loss':[]}

# # Keep track of train and evaluate perplexity.
# # This is a metric useful to track for language models.
# perplexity_history = {'train_perplexity':[], 'eval_perplexity':[]}


# import math
# # Keep track of train and evaluate loss.
# loss_history = {'train_loss':[]}

# # Keep track of train and evaluate perplexity.
# # This is a metric useful to track for language models.
# perplexity_history = {'train_perplexity':[]}
# # Loop through each log history.
# for log_history in trainer.state.log_history:

#   if 'loss' in log_history.keys():
#     # Deal with trianing loss.
#     loss_history['train_loss'].append(log_history['loss'])
#     perplexity_history['train_perplexity'].append(math.exp(log_history['loss']))
    
#   elif 'eval_loss' in log_history.keys():
#     # Deal with eval loss.
#     loss_history['eval_loss'].append(log_history['eval_loss'])
#     perplexity_history['eval_perplexity'].append(math.exp(log_history['eval_loss']))

# with open("output.txt","w") as f:
#     f.write("Perplexity")
#     f.write(perplexity_history)
#     f.write("loss history")
#     f.write(loss_history)


# # Plot Losses.
# plot_dict(loss_history, start_step=training_args.logging_steps, 
#           step_size=training_args.logging_steps, use_title='Loss', 
#           use_xlabel='Train Steps', use_ylabel='Values', magnify=2)

# print()

# # Plot Perplexities.
# plot_dict(perplexity_history, start_step=training_args.logging_steps, 
#           step_size=training_args.logging_steps, use_title='Perplexity', 
#           use_xlabel='Train Steps', use_ylabel='Values', magnify=2)