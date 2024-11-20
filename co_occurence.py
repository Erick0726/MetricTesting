from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load pre-trained BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define sample text corpus
corpus = """
The doctor treated the patient well. The man was a skilled surgeon.
The nurse was very kind. The woman assisted the nurse with great care.
He is a great doctor. She is a compassionate nurse.
The man was recognized as the best surgeon. The woman received praise as a dedicated nurse.
"""

# Define gendered terms and occupation-related terms
gendered_terms = ['man', 'woman', 'he', 'she']
occupations = ['doctor', 'nurse', 'surgeon']

# Preprocess the corpus: Convert to lowercase and split into words
corpus = corpus.lower()
sentences = re.split(r' *[\.\?!][\'"\)\]]* *', corpus)  # Split corpus into sentences


# Function to get BERT embeddings for a word in a sentence
def get_bert_embedding(sentence, word, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.tokenize(sentence)

    if word not in tokens:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings for the word from the BERT outputs
    word_index = tokens.index(word)
    return outputs.last_hidden_state[:, word_index, :].squeeze().numpy()


# Function to compute similarity score
def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]


# Initialize counters for co-occurrences and embeddings
cooccurrence_embeddings = {occupation: {'male': [], 'female': []} for occupation in occupations}

# Check for co-occurrences of gendered terms and occupations in each sentence
for sentence in sentences:
    for occupation in occupations:
        if occupation in sentence:
            for gender in gendered_terms:
                if gender in sentence:
                    # Get embeddings for the occupation and gender term
                    gender_embedding = get_bert_embedding(sentence, gender, model, tokenizer)
                    occupation_embedding = get_bert_embedding(sentence, occupation, model, tokenizer)

                    if gender_embedding is not None and occupation_embedding is not None:
                        # Compute the cosine similarity between gender and occupation embeddings
                        similarity = compute_similarity(gender_embedding, occupation_embedding)

                        # Classify by gender group
                        if gender in ['man', 'he']:
                            cooccurrence_embeddings[occupation]['male'].append(similarity)
                        elif gender in ['woman', 'she']:
                            cooccurrence_embeddings[occupation]['female'].append(similarity)

# Calculate the average bias score for each occupation
bias_scores = {}
for occupation, scores in cooccurrence_embeddings.items():
    male_scores = scores['male']
    female_scores = scores['female']

    avg_male_sim = np.mean(male_scores) if male_scores else 0
    avg_female_sim = np.mean(female_scores) if female_scores else 0

    # Bias score is the difference between male and female similarities
    bias_scores[occupation] = avg_male_sim - avg_female_sim

# Display the bias scores
for occupation, score in bias_scores.items():
    print(f"Bias score for '{occupation}': {score:.2f}")
