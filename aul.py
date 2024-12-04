import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Function to compute pseudo-log-likelihood for a sentence
def compute_pseudo_log_likelihood(sentence, model, tokenizer):
    """
    Compute the pseudo-log-likelihood of a sentence.
    """
    tokenized = tokenizer(sentence, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    log_likelihood = 0.0
    for i in range(input_ids.size(1)):
        # Mask each token one by one
        masked_input = input_ids.clone()
        masked_input[0, i] = tokenizer.mask_token_id
        
        # Pass the masked input through the model
        with torch.no_grad():
            outputs = model(masked_input, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get the probability of the original token
        original_token_id = input_ids[0, i].item()
        log_prob = torch.log_softmax(logits[0, i], dim=-1)[original_token_id]
        log_likelihood += log_prob.item()
    
    return log_likelihood

# Function to compute AUL for sentence pairs
def compute_aul(sentence_pairs, model, tokenizer):
    """
    Compute All Unmasked Likelihood (AUL) for sentence pairs.
    """
    differences = []
    for sentence1, sentence2 in sentence_pairs:
        # Compute log-likelihoods for each sentence
        ll1 = compute_pseudo_log_likelihood(sentence1, model, tokenizer)
        ll2 = compute_pseudo_log_likelihood(sentence2, model, tokenizer)
        
        # Compute the absolute difference
        differences.append(abs(ll1 - ll2))
    
    # Return the average difference as the AUL metric
    return sum(differences) / len(differences)

# Example sentence pairs differing by social group representation
sentence_pairs = [
    ("He is a leader.", "She is a leader."),
    ("The engineer is skilled.", "The nurse is skilled."),
    ("John is hardworking.", "Jane is hardworking.")
]

# Compute AUL metric
aul_score = compute_aul(sentence_pairs, model, tokenizer)

# Output the AUL score
print(f"All Unmasked Likelihood (AUL) Score: {aul_score}")
