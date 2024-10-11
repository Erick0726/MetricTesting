!pip install fairseq
!pip install numpy pandas scikit-learn
!pip install sacremoses

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the tokenizer and the model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de")

# Set model to evaluation mode (no gradients computation)
model.eval()

import pandas as pd

# Example dataset with sentences, labels, and demographic information (gender)
data = {
    'sentence': ["This is great!", "I don't like it.", "Amazing product!", "Not a fan.", "This is perfect!"],
    'label': [1, 0, 1, 0, 1],  # 1 = positive, 0 = negative
    'gender': ['male', 'female', 'male', 'female', 'male']
}

df = pd.DataFrame(data)

def get_model_probabilities(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the probability scores (logits) and convert to softmax probabilities
        probs = torch.softmax(outputs.logits, dim=-1)
    # Assuming you want the probability of the first token in the output sequence
    # being the second token in the vocabulary, you would use [0, 0, 1]
    # Adjust the indices as needed based on your specific use case
    return probs[0, 0, 1].item()  # Changed here

# Add probability predictions to the dataframe
df['probability'] = df['sentence'].apply(get_model_probabilities)

import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_lift_curve(true_labels, predicted_probs, group):
    # Sort by predicted probabilities in descending order
    sorted_indices = np.argsort(predicted_probs)[::-1]
    sorted_labels = np.array(true_labels)[sorted_indices]

    # Calculate the cumulative true positives (TP)
    cumulative_true_positives = np.cumsum(sorted_labels)

    # Calculate lift: ratio of TP to the baseline rate
    baseline_rate = np.mean(true_labels)
    lift = cumulative_true_positives / (np.arange(1, len(sorted_labels) + 1) * baseline_rate)

    return lift

def calculate_AUL(df, group_column):
    groups = df[group_column].unique()
    aul_scores = {}

    for group in groups:
        group_df = df[df[group_column] == group]
        true_labels = group_df['label'].values
        predicted_probs = group_df['probability'].values

        # Calculate the Lift curve and area under the lift curve (AUL)
        lift_curve = compute_lift_curve(true_labels, predicted_probs, group)
        aul = np.trapz(lift_curve) / len(lift_curve)

        aul_scores[group] = aul

    return aul_scores

# Compute AUL scores for gender groups
aul_scores = calculate_AUL(df, 'gender')
print("AUL Scores by Gender:", aul_scores)