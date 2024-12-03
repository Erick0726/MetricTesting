import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load pre-trained Sentence-BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the JSON data from a file
with open('tests/seat.json', 'r') as f:
    data = json.load(f)

# Extract target and attribute sentences from the JSON file
targ1_sentences = data['targ1']['examples']
targ2_sentences = data['targ2']['examples']
attr1_sentences = data['attr1']['examples']
attr2_sentences = data['attr2']['examples']

# Convert sentences to embeddings using Sentence-BERT
def get_sentence_embeddings(sentences, model):
    return model.encode(sentences)

# Function to compute the SEAT metric
def seat_metric(target_s1, target_s2, attribute_s1, attribute_s2):
    def s(w, X):
        return np.mean([1 - cosine(w, x) for x in X])

    t1a1 = np.mean([s(w, attribute_s1) for w in target_s1])
    t1a2 = np.mean([s(w, attribute_s2) for w in target_s1])
    t2a1 = np.mean([s(w, attribute_s1) for w in target_s2])
    t2a2 = np.mean([s(w, attribute_s2) for w in target_s2])

    return t1a1 - t1a2 - (t2a1 - t2a2)

# Generate embeddings for target and attribute sentences
targ1_embeddings = get_sentence_embeddings(targ1_sentences, model)
targ2_embeddings = get_sentence_embeddings(targ2_sentences, model)
attr1_embeddings = get_sentence_embeddings(attr1_sentences, model)
attr2_embeddings = get_sentence_embeddings(attr2_sentences, model)

# Calculate SEAT score
seat_score = seat_metric(targ1_embeddings, targ2_embeddings, attr1_embeddings, attr2_embeddings)
print(f'White Female Names vs Black Female Names SEAT score: {seat_score}')
