import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import numpy as np

model1_name = 'bert-base-uncased'
model1 = AutoModelForMaskedLM.from_pretrained(model1_name)
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
model2_name = 'roberta-base'
model2 = AutoModelForMaskedLM.from_pretrained(model2_name)
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model3_name = 'albert-base-v2'
model3 = AutoModelForMaskedLM.from_pretrained(model3_name)
tokenizer3 = AutoTokenizer.from_pretrained(model3_name)

crows_pairs_file = 'crows_pairs_anonymized.csv'
crows_pairs = pd.read_csv(crows_pairs_file)

total = len(crows_pairs)

def get_sentence_probability(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return np.exp(-loss)

def get_cps(model, tokenizer):
    biased_higher_count = 0
    for _, row in crows_pairs.iterrows():
        biased_sentence = row['sent_more']
        neutral_sentence = row['sent_less']
        biased_prob = get_sentence_probability(biased_sentence, model, tokenizer)
        neutral_prob = get_sentence_probability(neutral_sentence, model, tokenizer)
        if biased_prob > neutral_prob:
            biased_higher_count += 1
    bias_score = biased_higher_count / total
    return bias_score

bert_bias_score = get_cps(model1, tokenizer1)
roberta_bias_score = get_cps(model2, tokenizer2)
albert_bias_score = get_cps(model3, tokenizer3)

print(f"CrowS-Pairs bias score for BERT: {bert_bias_score}")
print(f"CrowS-Pairs bias score for RoBERTa: {roberta_bias_score}")
print(f"CrowS-Pairs bias score for ALBERT: {albert_bias_score}")