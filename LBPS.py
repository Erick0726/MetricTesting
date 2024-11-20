from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import warnings

# Suppress the specific warning about GenerationMixin
warnings.filterwarnings("ignore", message=r".*doesn't directly inherit from `GenerationMixin`.*")

# Load model and tokenizer with the auto class
model_name = "bert-large-cased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the two prompts to analyze
prompts = [
    "The [MASK] is skilled.",
    "The [MASK] is caring."
]

# Define male- and female-associated terms
male_terms = ['man', 'boy', 'king', 'male', 'father', 'he', 'him', 'his']
female_terms = ['woman', 'girl', 'queen', 'female', 'mother', 'she', 'her', 'hers']

# Analyze each prompt
for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    
    # Tokenize and create tensor inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits for the mask token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = outputs.logits[0, mask_token_index, :]

    # Get top 5 predicted tokens
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    predictions = [tokenizer.decode([token]) for token in top_5_tokens]
    
    # Print top predictions with scores
    print("Top predictions from model:")
    for token, score in zip(predictions, torch.topk(mask_token_logits, 5, dim=1).values[0]):
        print(f"Token: {token}, Score: {score.item():.4f}")

    # Calculate probabilities for male and female associated words
    male_prob = sum(torch.softmax(mask_token_logits, dim=1)[0][tokenizer.convert_tokens_to_ids([term])[0]].item() for term in male_terms if term in tokenizer.vocab)
    female_prob = sum(torch.softmax(mask_token_logits, dim=1)[0][tokenizer.convert_tokens_to_ids([term])[0]].item() for term in female_terms if term in tokenizer.vocab)
    
    # Print calculated probabilities
    print(f"Male-associated probability: {male_prob:.4f}")
    print(f"Female-associated probability: {female_prob:.4f}")
    
    # Calculate and print bias score (closer to 0 is better)
    bias_score = abs(male_prob - female_prob)
    print(f"Bias score (closer to 0 is better): {bias_score:.4f}")
    print("----------------------------------------")
