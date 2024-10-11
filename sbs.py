# Import necessary libraries
from transformers import XLNetTokenizer, XLNetModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')

# Function to get the embedding of a word
def get_embedding(word):
    # Tokenize the input word
    tokens = tokenizer(word, return_tensors='pt')
    # Pass the tokenized input to the model and get the last hidden state
    with torch.no_grad():
        outputs = model(**tokens)
    # Mean pooling of the hidden states to get a single embedding for the word
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Define target words (e.g., gendered or race-related words) and attribute words (e.g., occupations)
target_words = ["man", "woman", "Black", "white"]
attribute_words = ["doctor", "nurturer", "leader", "scientist"]

# Initialize a dictionary to store embeddings
embeddings = {}

# Calculate embeddings for each target word
for word in target_words:
    embeddings[word] = get_embedding(word)

# Calculate embeddings for each attribute word
for word in attribute_words:
    embeddings[word] = get_embedding(word)

# Function to compute cosine similarity between two words
def compute_similarity(word1, word2):
    return cosine_similarity(embeddings[word1], embeddings[word2])[0][0]

# Compute similarities between each target word and attribute word
similarity_scores = {}

# Loop through each pair of target and attribute words
for target in target_words:
    similarity_scores[target] = {}
    for attribute in attribute_words:
        score = compute_similarity(target, attribute)
        similarity_scores[target][attribute] = score

# Display the cosine similarity results
print("Cosine Similarity Scores between Target Words and Attribute Words:\n")
for target in similarity_scores:
    for attribute in similarity_scores[target]:
        print(f"Similarity between '{target}' and '{attribute}': {similarity_scores[target][attribute]:.4f}")

# (Optional) Convert results to a matrix or other data format for further analysis
import pandas as pd

# Create a DataFrame for easier visualization of the results
similarity_df = pd.DataFrame(similarity_scores)

# Display the DataFrame in a tabular format
print("\nCosine Similarity Matrix:")
print(similarity_df)
