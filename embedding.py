"""
@author sourabhxiii
"""

import re
import io
import sys
import os
import gc
import operator 
import numpy as np
from tqdm import tqdm
import gensim.models as gsm

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

# GLOVE_FOLDER = r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\embeddings\glove.6B'
# FASTEXT_FOLDER = r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\embeddings\fasttext'
# PARAGRAM_FOLDER = r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\embeddings\paragram'
# EMOJI2VEC_FOLDER = r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\embeddings\emoji2vec'
GLOVE_FOLDER = r'/home/dell/sm186047/embeddings/glove.6B'
FASTEXT_FOLDER = r'/home/dell/sm186047/embeddings/fasttext'
PARAGRAM_FOLDER = r'/home/dell/sm186047/embeddings/paragram'
EMOJI2VEC_FOLDER = r'/home/dell/sm186047/embeddings/emoji2vec'


# e2v['\U0001f604']

class Embedding:
    def __init__(self, word_index):
        self.max_features = len(word_index)+1
        self.word_index = word_index
        self.e2v = \
            gsm.KeyedVectors.load_word2vec_format(EMOJI2VEC_FOLDER+os.sep+'emoji2vec.bin'
                , binary=True)

    def _get_embedding_index(self, embedding_file):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if o.split(" ")[0] in self.word_index)
        return embedding_index

    def _load_glove(self):
        EMBEDDING_FILE = GLOVE_FOLDER+os.sep+'glove.6B.300d.txt'
        # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8") if o.split(" ")[0] in self.word_index)
        embeddings_index = self._get_embedding_index(EMBEDDING_FILE)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.max_features, embed_size))
        for word, i in tqdm(self.word_index.items()):
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            else:
                try:
                    embedding_vector = self.e2v[word]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                except KeyError as _:
                    pass
                
        return embedding_matrix 
        
    def _load_fasttext(self):    
        EMBEDDING_FILE = FASTEXT_FOLDER+os.sep+'crawl-300d-2M.vec'+os.sep+'crawl-300d-2M.vec'
        # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8") if len(o)>100 and o.split(" ")[0] in self.word_index )
        embeddings_index = self._get_embedding_index(EMBEDDING_FILE)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.max_features, embed_size))
        for word, i in tqdm(self.word_index.items()):
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            else:
                try:
                    embedding_vector = self.e2v[word]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                except KeyError as _:
                    pass

        return embedding_matrix

    def _load_para(self):
        EMBEDDING_FILE = PARAGRAM_FOLDER+os.sep+'paragram_300_sl999'+os.sep+'paragram_300_sl999.txt'
        # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in self.word_index)
        embeddings_index = self._get_embedding_index(EMBEDDING_FILE)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        
        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.max_features, embed_size))
        for word, i in tqdm(self.word_index.items()):
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            else:
                try:
                    embedding_vector = self.e2v[word]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                except KeyError as _:
                    pass

        return embedding_matrix

    def get_embedding_matrix(self):
        print('Preparing embedding matrix.')
        embedding_matrix_1 = self._load_glove()
        embedding_matrix_2 = self._load_fasttext()
        embedding_matrix_3 = self._load_para()
        embedding_matrix = np.mean(
                (embedding_matrix_1, embedding_matrix_2, embedding_matrix_3)
                , axis=0)  
        del embedding_matrix_1, embedding_matrix_3
        gc.collect()
        print('Shape of embedding matrix: %s' % str(embedding_matrix.shape))
        return embedding_matrix
    

    def get_embedding_vocab(self):
        print('Preparing embedding vocab.')
        EMBEDDING_FILE = GLOVE_FOLDER+os.sep+'glove.6B.300d.txt'
        g_embeddings_index = self._get_embedding_index(EMBEDDING_FILE)
        EMBEDDING_FILE = FASTEXT_FOLDER+os.sep+'crawl-300d-2M.vec'+os.sep+'crawl-300d-2M.vec'
        f_embeddings_index = self._get_embedding_index(EMBEDDING_FILE)
        EMBEDDING_FILE = PARAGRAM_FOLDER+os.sep+'paragram_300_sl999'+os.sep+'paragram_300_sl999.txt'
        p_embeddings_index = self._get_embedding_index(EMBEDDING_FILE)
        e_embeddings_index = self.e2v.vocab

        all_words = set().union(*[g_embeddings_index, f_embeddings_index, p_embeddings_index, e_embeddings_index])
        return all_words

    def check_coverage(self, docu_vocab, embedding_vocab):
        print('Checking document vocab coverage in embedding.')
        from collections import Counter
        in_embedding_vocab = Counter()
        oov = Counter()
        covered_word_count = 0
        oov_word_count = 0
        for word in tqdm(docu_vocab):
            if word in embedding_vocab:
                in_embedding_vocab[word] = docu_vocab[word]
                covered_word_count += docu_vocab[word]
            else:
                oov[word] = docu_vocab[word]
                oov_word_count += docu_vocab[word]
                pass

        print('Embedding was not found for %d unique words with total occurence %d.' % (len(oov), oov_word_count))
        print('Found embeddings for {:.2%} of vocab'.format(len(in_embedding_vocab) / len(docu_vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(covered_word_count / (covered_word_count + oov_word_count)))

        return oov
