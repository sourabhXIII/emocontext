"""
@author sourabhxiii
"""
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

class Vectorize:
    def __init__(self, sentences, MAX_VOCAB_SIZE):
        
        self.max_vocab_size = MAX_VOCAB_SIZE
        self.num_words = 0
        # load data file
        self.sentences = sentences

        # get a tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, filters=[])
        # tokenize
        self.tokenizer.fit_on_texts(self.sentences)
        
        # get word -> integer mapping
        self.word2idx = self.tokenizer.word_index
        self.idx2word = dict(map(reversed, self.tokenizer.word_index.items()))
        self.word_counts = self.tokenizer.word_counts

        # glimpse of data
        self.describe()
        return

    def describe(self):
        print('Found %s unique tokens.' % len(self.word2idx))
        print("max sequence length:", max(len(s) for s in self.sentences))
        print("min sequence length:", min(len(s) for s in self.sentences))
        s = sorted(len(s) for s in self.sentences)
        print("median sequence length:", s[len(s) // 2])

    def set_num_words(self, num_words):
        self.num_words = num_words

    def vectorize_data(self, sentences, max_sequence_length):
        sequences = self.tokenizer.texts_to_sequences(sentences)
        # pad sequences so that we get a N x T matrix
        data = pad_sequences(sequences
                , maxlen=max_sequence_length)
        print('Shape of data tensor:', data.shape)
        return data
    
    def get_test_data(self, texts, max_sequence_length):
        seq = self.tokenizer.texts_to_sequences(texts)
        data = pad_sequences(seq, maxlen=max_sequence_length)
        return data

    def save_tokenizer(self, file_path):
        import pickle
        
        with open(file_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return