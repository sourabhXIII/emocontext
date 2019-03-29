"""
@author sourabhxiii
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import class_weight

from keras.utils import to_categorical

tqdm.pandas()

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
OUTPUT_DIM = 4
kfold_splits = 5

BASE_PATH = r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\emo_context'
TOKENIZER_PATH = BASE_PATH+os.sep+r'tokenizer.pcl'
TRAIN_DATA_PATH = BASE_PATH+os.sep+r'data'+os.sep+'train_combined.txt'

TEST_DATA_PATH = BASE_PATH+os.sep+r'data'+os.sep+r'test.txt'

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def load_data(filepath, mode='train'):
    from preprocess import preprocessData
    if mode == 'train':
        indices, conversations, labels = preprocessData(filepath, emotion2label, mode)
        print('Training data shape: %s' % str(len(conversations)))
        return indices, conversations, labels
    else:
        indices, conversations = preprocessData(filepath, emotion2label, mode)
        return indices, conversations

def check_embedding_quality(conversations):
    from vectorize import Vectorize
    vectorizer = Vectorize(conversations, MAX_VOCAB_SIZE)
    word_index = vectorizer.word2idx

    from embedding import Embedding
    embed = Embedding(word_index)

    docu_vocab = vectorizer.word_counts
    embedding_vocab = embed.get_embedding_vocab()
    oov_words = embed.check_coverage(docu_vocab, embedding_vocab)
    print('Collected oov words.')
    return oov_words

def get_embedding_matrix_and_vectorizer(conversations):
    from vectorize import Vectorize
    vectorizer = Vectorize(conversations, MAX_VOCAB_SIZE)
    word_index = vectorizer.word2idx
    # train_sequences = vectorizer.vectorize_data(conversations, MAX_SEQUENCE_LENGTH)
    vectorizer.save_tokenizer(TOKENIZER_PATH)

    from embedding import Embedding
    embed = Embedding(word_index)
    embedding_matrix = embed.get_embedding_matrix()

    return embedding_matrix, vectorizer

def get_train_sequences(vectorizer, conversations):
    train_sequences = vectorizer.vectorize_data(conversations, MAX_SEQUENCE_LENGTH)
    return train_sequences

# def train_model(embedding_matrix, conversations, targets, class_weights):
def train_model(embedding_matrix, turn1_sequences, turn2_sequences, turn3_sequences
        , turn1_woe, turn2_woe, turn3_woe
        , targets, class_weights):
    
    from sklearn.model_selection import train_test_split
    t1_train, t1_test, t2_train, t2_test, t3_train, t3_test, t1_woe_train, t1_woe_test \
        , t2_woe_train, t2_woe_test, t3_woe_train, t3_woe_test, target_train, target_test = \
        train_test_split(turn1_sequences, turn2_sequences, turn3_sequences
            , turn1_woe, turn2_woe, turn3_woe
            , targets, stratify=targets
            , test_size=0.1, random_state=42)

    # build model
    # from conditioned_classifier import SimpleClassifier
    # emocon = SimpleClassifier(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, OUTPUT_DIM)
    # from classifier import EmoconModel
    # emocon = EmoconModel(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, OUTPUT_DIM)

    # from conditioned_classifier import ConditionedModelV1
    # emocon = ConditionedModelV1(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, OUTPUT_DIM)
    
    # from conditioned_classifier import TextualFeatureModel
    # emocon = TextualFeatureModel(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, OUTPUT_DIM)
    

    from conditioned_classifier import ConditionedModelV2
    emocon = ConditionedModelV2(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, OUTPUT_DIM)
    # train model
    hist = emocon.train(t1_train, t1_test, t2_train, t2_test, t3_train, t3_test, t1_woe_train, t1_woe_test
        , t2_woe_train, t2_woe_test, t3_woe_train, t3_woe_test, target_train, target_test, class_weights)
    # hist = emocon.train(conv_train, conv_test, target_train, target_test, class_weights)
    return hist

def resume_trainig(model_path, initial_epoch, turn1, turn2, turn3, targets, class_weights):
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import load_model

    # load vectorizer
    import pickle
    vectorizer = None
    with open(TOKENIZER_PATH, 'rb') as handle:
        vectorizer = pickle.load(handle)
    
    # prepare test sequences
    turn1_sentences = vectorizer.texts_to_sequences(turn1)
    turn2_sentences = vectorizer.texts_to_sequences(turn2)
    turn3_sentences = vectorizer.texts_to_sequences(turn3)
    # pad sequences so that we get a N x T matrix
    turn1_sequences = pad_sequences(turn1_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    turn2_sequences = pad_sequences(turn2_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    turn3_sequences = pad_sequences(turn3_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of resume train data tensor:', turn1_sequences.shape)

    from preprocess import get_woe_vector
    woe_arr = get_woe_vector(TRAIN_DATA_PATH, mode='train')
    turn1_woe = woe_arr[:,0, :]
    turn2_woe = woe_arr[:,1, :]
    turn3_woe = woe_arr[:,2, :]
    print('Collected woe array.')

    from sklearn.model_selection import train_test_split
    t1_train, t1_test, t2_train, t2_test, t3_train, t3_test, t1_woe_train, t1_woe_test \
        , t2_woe_train, t2_woe_test, t3_woe_train, t3_woe_test, target_train, target_test = \
        train_test_split(turn1_sequences, turn2_sequences, turn3_sequences
            , turn1_woe, turn2_woe, turn3_woe
            , targets, stratify=targets
            , test_size=0.1, random_state=42)

    BATCH_SIZE = 128
    EPOCHS = 100

    # set up callbacks
    filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
    chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
        , save_best_only=True, save_weights_only=False, mode='auto', period=1)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
        , verbose=1, mode='auto', baseline=None)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
        patience=2, min_lr=0.0001, verbose=1)

    callback_list=[chkpoint]

    model = load_model(model_path)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(0.001),
            metrics=['acc']
        )

    hist = model.fit(
            [t1_train, t2_train, t3_train, t1_woe_train, t2_woe_train, t3_woe_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([t1_test, t2_test, t3_test, t1_woe_test, t2_woe_test, t3_woe_test], target_test),
            verbose=1,
            class_weight=class_weights,
            callbacks=callback_list,
            shuffle=True,
            initial_epoch=initial_epoch
            )

    return hist

def test_model(model_path, filepath):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import load_model
    
    indices, conversations = load_data(filepath, mode='test')
    from preprocess import split_turns
    turn_arr = split_turns(conversations, 3)
    print("Broken conversations into turns.")

    # load vectorizer
    import pickle
    vectorizer = None
    with open(TOKENIZER_PATH, 'rb') as handle:
        vectorizer = pickle.load(handle)
    
    # prepare test sequences
    turn1_sentences = vectorizer.texts_to_sequences(turn_arr[:, 0].tolist())
    turn2_sentences = vectorizer.texts_to_sequences(turn_arr[:, 1].tolist())
    turn3_sentences = vectorizer.texts_to_sequences(turn_arr[:, 2].tolist())
    # pad sequences so that we get a N x T matrix
    turn1_sequences = pad_sequences(turn1_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    turn2_sequences = pad_sequences(turn2_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    turn3_sequences = pad_sequences(turn3_sentences
                        , maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of test data tensor:', turn1_sequences.shape)

    from preprocess import get_woe_vector
    woe_arr = get_woe_vector(filepath, mode='test')
    turn1_woe = woe_arr[:,0, :]
    turn2_woe = woe_arr[:,1, :]
    turn3_woe = woe_arr[:,2, :]
    print('Collected woe array.')

    # load trained model
    model = load_model(model_path)
    print(model.summary())
    # make predction
    # sequences = vectorizer.texts_to_sequences(conversations)
    # sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # predictions = model.predict(sequences, verbose=1 )
    predictions = model.predict(
        [turn1_sequences, turn2_sequences, turn3_sequences, turn1_woe, turn2_woe, turn3_woe], verbose=1)
    predictions = predictions.argmax(axis=1)

    return predictions

def create_submission_file(test_data_path, solution_path, predictions):
    import io
    
    with io.open(solution_path, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(test_data_path, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')

def main():
    indices, conversations, labels = load_data(TRAIN_DATA_PATH, mode='train')
    # indices = indices[:1024]
    # conversations = conversations[:1024]
    # labels = labels[:1024]

    from preprocess import split_turns
    turn_arr = split_turns(conversations, 3)
    print("Broken conversations into turns.")

    turn_list = []
    turn_list.extend(turn_arr[:, 0].tolist())
    turn_list.extend(turn_arr[:, 1].tolist())
    turn_list.extend(turn_arr[:, 2].tolist())

    # # oov_words = check_embedding_quality(turn_list)

    # # embedding_matrix, vectorizer = \
    # #     get_embedding_matrix_and_vectorizer(conversations)
    # # conv_sequences = get_train_sequences(vectorizer, conversations)


    embedding_matrix, vectorizer = \
        get_embedding_matrix_and_vectorizer(turn_list)
    turn1_sequences = get_train_sequences(vectorizer, turn_arr[:, 0].tolist())
    turn2_sequences = get_train_sequences(vectorizer, turn_arr[:, 1].tolist())
    turn3_sequences = get_train_sequences(vectorizer, turn_arr[:, 2].tolist())
    targets = to_categorical(np.asarray(labels))
    print('Collected embedding matrix and sequenced training data.')

    from preprocess import get_woe_vector
    woe_arr = get_woe_vector(TRAIN_DATA_PATH, mode='train')
    turn1_woe = woe_arr[:,0, :]
    turn2_woe = woe_arr[:,1, :]
    turn3_woe = woe_arr[:,2, :]
    print('Collected woe array.')

    class_weights = \
        class_weight.compute_sample_weight('balanced', np.unique(labels), labels)
    print('Using smaple weight: %s' % str(class_weights))

    # # hist = train_model(embedding_matrix
    # #     , conv_sequences, targets, class_weights)

    hist = train_model(embedding_matrix
        , turn1_sequences, turn2_sequences, turn3_sequences
        , turn1_woe, turn2_woe, turn3_woe
        , targets, class_weights)

    
    # initial_epoch=50
    # hist = resume_trainig(BASE_PATH+os.sep+r'model.50-0.4004-0.4039.hdf5'
    #     , initial_epoch
    #     , turn_arr[:, 0].tolist(), turn_arr[:, 1].tolist(), turn_arr[:, 2].tolist()
    #     , targets, class_weights)

    # print(hist.history)

    # predictions = test_model(BASE_PATH+os.sep+r'model.60-0.3835-0.3824.hdf5', TEST_DATA_PATH)
    # create_submission_file(TEST_DATA_PATH
    #     , BASE_PATH+os.sep+r'answer'+os.sep+r'test.txt'
    #     , predictions)

    print('Done!!')


if __name__ == '__main__':
    main()
