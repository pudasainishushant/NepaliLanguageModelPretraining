## NepBERT

![NEPBERT](nepBERT.png "NEPBERT")

### Purpose  Train a custom language model for Nepali dataset to generate proper word embedding for Nepali text

## Features
- Trained on more than 200 million of Nepali sentences from Nepali news dataset scrapped from several Nepali news websites
- Text perplexity score of 60.78 on evaluation dataset containing more than 3000 Nepali sentences
- Trained using NVIDIA-RTX 2080 for 3 days

## Usage

- First install transformers in your device using 
```console
foo@bar:~$ pip install transformers
```

```

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("Shushant/NepNewsBERT")

model = AutoModelForMaskedLM.from_pretrained("Shushant/NepNewsBERT")

from transformers import pipeline

fill_mask = pipeline( "fill-mask", model=model, tokenizer=tokenizer, ) 
from pprint import pprint pprint(fill_mask(f"तिमीलाई कस्तो {tokenizer.mask_token}."))

```

# Paper Details

If you are interested in this research, please read the full paper here.
https://www.researchgate.net/publication/375019515_NepaliBERT_Pre-training_of_Masked_Language_Model_in_Nepali_Corpus/citations

## Citation Plain Text
S. Pudasaini, S. Shakya, A. Tamang, S. Adhikari, S. Thapa and S. Lamichhane, "NepaliBERT: Pre-training of Masked Language Model in Nepali Corpus," 2023 7th International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC), Kirtipur, Nepal, 2023, pp. 325-330, doi: 10.1109/I-SMAC58438.2023.10290690.

## Citation Bibtex
S. Pudasaini, S. Shakya, A. Tamang, S. Adhikari, S. Thapa and S. Lamichhane, "NepaliBERT: Pre-training of Masked Language Model in Nepali Corpus," 2023 7th International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC), Kirtipur, Nepal, 2023, pp. 325-330, doi: 10.1109/I-SMAC58438.2023.10290690.
