import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import gensim.downloader

def weat_score(file_name):
    f = open(file_name)
    data = json.load(f)
    #target 1
    A = data["targ1"]["examples"]
    #target 2
    B = data["targ2"]["examples"]
    #attribute 1
    X = data["attr1"]["examples"]
    #attribute 2
    Y = data["attr2"]["examples"]
    f.close()

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

    weat_score_glove = sum([s(a.lower(), X, Y, word_vectors) for a in A]) - sum ([s(b.lower(), X, Y, word_vectors) for b in B])
    print(f'WEAT score for GloVe ({data["targ1"]["category"]}/{data["targ2"]["category"]}/{data["attr1"]["category"]}/{data["attr2"]["category"]}): {weat_score_glove}')

    #To run the test for the Word2Vec model, download the file from this link and put the GoogleNews-vectors-negative300.bin
    #file in the same folder as the weat.py file:
    #https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g
    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #weat_score_word2vec = sum([s(a.lower(), X, Y, model) for a in A]) - sum ([s(b.lower(), X, Y, model) for b in B])
    #print(f'WEAT score for Word2Vec ({data["targ1"]["category"]}/{data["targ2"]["category"]}/{data["attr1"]["category"]}/{data["attr2"]["category"]}): {weat_score_word2vec}')

weat_score("tests/weat1.jsonl")
weat_score("tests/weat2.jsonl")
weat_score("tests/weat3.jsonl")
