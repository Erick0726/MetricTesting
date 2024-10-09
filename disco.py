import numpy as np
from numpy.linalg import norm


# Load GloVe model from text file
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Loaded {len(embeddings_index)} word vectors.')
    return embeddings_index


# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# Compute DisCo metric between two words
def disco_metric(word1, word2, embeddings):
    if word1 not in embeddings or word2 not in embeddings:
        print(f"One of the words '{word1}' or '{word2}' is not in the GloVe vocabulary.")
        return None

    word1_vec = embeddings[word1]
    word2_vec = embeddings[word2]

    # Compositional vector by adding word vectors
    composite_vec = word1_vec + word2_vec

    # Check similarities
    word1_sim = cosine_similarity(composite_vec, word1_vec)
    word2_sim = cosine_similarity(composite_vec, word2_vec)

    # DisCo score is the average of the similarities
    disco_score = (word1_sim + word2_sim) / 2
    return disco_score


# Load GloVe embeddings (make sure to have 'glove.6B.300d.txt' in the same directory)
glove_embeddings = load_glove_embeddings('glove.6B.300d.txt')

# Testing DisCo metric with gendered terms and professions
test_pairs = [
    ("man", "doctor"),
    ("woman", "doctor"),
    ("man", "nurse"),
    ("woman", "nurse"),
    ("man", "scientist"),
    ("woman", "scientist")
]

# Compute and display DisCo scores for each pair
for pair in test_pairs:
    word1, word2 = pair
    score = disco_metric(word1, word2, glove_embeddings)
    if score:
        print(f"DisCo score for '{word1}' and '{word2}': {score:.4f}")
