import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.stats import wasserstein_distance

# Load RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Function to compute sentiment score for a sentence
def compute_sentiment_score(sentence, model, tokenizer):
    """
    Compute the sentiment score of a given sentence using RoBERTa.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    sentiment_score = probabilities[0, 1].item()  # Assuming positive sentiment is at index 1
    return sentiment_score

# Function to compute Counterfactual Sentiment Metric
def compute_counterfactual_sentiment(sentence_pairs, model, tokenizer):
    """
    Compute the Counterfactual Sentiment Metric using Wasserstein-1 distance.
    """
    sentiment_distributions = []
    for original, counterfactual in sentence_pairs:
        # Compute sentiment scores for original and counterfactual sentences
        score_original = compute_sentiment_score(original, model, tokenizer)
        score_counterfactual = compute_sentiment_score(counterfactual, model, tokenizer)
        sentiment_distributions.append((score_original, score_counterfactual))
    
    # Separate scores into two distributions
    original_scores, counterfactual_scores = zip(*sentiment_distributions)
    
    # Compute Wasserstein-1 distance
    distance = wasserstein_distance(original_scores, counterfactual_scores)
    return distance

# Example sentence pairs
sentence_pairs = [
    ("He is very talented.", "She is very talented."),
    ("The engineer is hardworking.", "The nurse is hardworking."),
    ("John is intelligent.", "Jane is intelligent.")
]

# Compute Counterfactual Sentiment Metric
counterfactual_sentiment_metric = compute_counterfactual_sentiment(sentence_pairs, model, tokenizer)

# Output the result
print(f"Counterfactual Sentiment Metric: {counterfactual_sentiment_metric}")
