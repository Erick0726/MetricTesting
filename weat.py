import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

#target 1
A = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
#target 2
B = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
#attribute 1
X = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']\
#attribute 2
Y = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']

def load_word_vectors_glove(glove_file):
    word_v = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_v[word] = vector
    return word_v

word_vectors = load_word_vectors_glove('glove.6B.50d.txt')

def s(w, X, Y, vectors):
    sim_X = np.mean([cosine_similarity(vectors[w].reshape(1,-1), vectors[x].reshape(1,-1)) for x in X])
    sim_Y = np.mean([cosine_similarity(vectors[w].reshape(1,-1), vectors[y].reshape(1,-1)) for y in Y])
    return sim_X - sim_Y

weat_score_glove = sum([s(a, X, Y, word_vectors) for a in A]) - sum ([s(b, X, Y, word_vectors) for b in B])
print(f'WEAT score for GloVe: {weat_score_glove}')

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
weat_score_word2vec = sum([s(a, X, Y, model) for a in A]) - sum ([s(b, X, Y, model) for b in B])
print(f'WEAT score for Word2Vec: {weat_score_word2vec}')
