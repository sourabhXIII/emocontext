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
import pickle

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def preprocessData(dataFilePath, emotion2label=emotion2label, mode='train', convert=False):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    import emoji

    print("Started preprocessing.")
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in tqdm(finput):
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4])
            conv = conv.lower()

            # add space around emojis
            try:
                # Wide UCS-4 build
                oRes = re.compile(u'(['
                    u'\U0001F300-\U0001F64F'
                    u'\U0001F680-\U0001F6FF'
                    u'\u2600-\u26FF\u2700-\u27BF]+)', 
                    re.UNICODE)
            except re.error:
                # Narrow UCS-2 build
                oRes = re.compile(u'(('
                    u'\ud83c[\udf00-\udfff]|'
                    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                    u'[\u2600-\u26FF\u2700-\u27BF])+)', 
                    re.UNICODE)
            conv = oRes.sub(r'  \1  ', conv)

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            # remove trailing 's
            conv = re.sub("’s", "", conv)
            conv = re.sub("'s", "", conv)
            # replace common abbreviations
            conv = re.sub("&apos;", "'", conv)
            conv = re.sub("n't", " not", conv)
            conv = re.sub("'ll", " will", conv)
            conv = re.sub("'ve", " have", conv)
            conv = re.sub("'re", " are", conv)
            conv = re.sub("'d", " would", conv)
            conv = re.sub("’m", " am", conv)
            conv = re.sub("'m", " am", conv)
            conv = re.sub(" u are", " you are", conv)
            conv = re.sub(" ur ", " your ", conv)
            conv = re.sub(" wanna ", " want to ", conv)
            conv = re.sub("haha", emoji.emojize(":beaming_face_with_smiling_eyes:"), conv)
            conv = re.sub("hehe", emoji.emojize(":beaming_face_with_smiling_eyes:"), conv)
            conv = re.sub("hah", emoji.emojize(":slightly_smiling_face:"), conv)
            conv = re.sub("hmm", emoji.emojize(":neutral_face:"), conv)
            # replace commonly misspelled words
            conv = re.sub("ok~~ay~~", "ok", conv)
            # appears so many times
            conv = re.sub(":/", emoji.emojize(' :slightly_frowning_face: '), conv)
            conv = re.sub(r"-\)", emoji.emojize(' :slightly_smiling_face: '), conv)
            conv = re.sub(r"\(", emoji.emojize(' :crying_face: '), conv)
            conv = re.sub(r"-\(", emoji.emojize(' :crying_face: '), conv)
            conv = re.sub(r"'-\(", emoji.emojize(' :crying_face: '), conv)
            conv = re.sub(r"\)\)", emoji.emojize(' :beaming_face_with_smiling_eyes: '), conv)
            conv = re.sub(r":P", emoji.emojize(' :winking_face_with_tongue: '), conv)

            if emoji.emoji_count(conv) :
                # greater than zero means emoji exists
                conv = emoji.emojize(re.sub('::', ': :', emoji.demojize(conv)))
            # re.sub(r"\s+$", "", (list(words)[0]+' ')*len(list(words)))

            if convert is True:
                # replace all the emojis with their text
                conv = re.sub(':', '', re.sub('::', '_', emoji.demojize(conv))).split('_')
                conv = " ".join(conv)

            conv = re.sub(";", "", conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())
    print("Done preprocessing.")
    
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def split_turns(long_convs, n_turns=3):
    n_long_convs = len(long_convs)
    
    conv_arr = np.ndarray(shape=(n_long_convs, n_turns), dtype=object)

    for c in range(n_long_convs):
        conv = long_convs[c]
        turns = conv.split(sep=" <eos> ")
        for t in range(n_turns):
            conv_arr[c][t] = turns[t]
    
    return conv_arr

def get_woe_per_word(happy_counts, sad_counts, angry_counts, others_counts, total_counts, n_top_words):
    word_woe = dict()
    top_happy = dict(happy_counts.most_common(n_top_words))
    top_sad = dict(sad_counts.most_common(n_top_words))
    top_angry = dict(angry_counts.most_common(n_top_words))
    top_others = dict(others_counts.most_common(n_top_words))

    for k, v in top_happy.items():
        a = top_angry[k] if k in top_angry.keys() else 0.0
        s = top_sad[k] if k in top_sad.keys() else 0.0
        o = top_others[k] if k in top_others.keys() else 0.0
        h_woe = np.tanh((v/total_counts[k])/(a+s+o+0.001))
        word_woe[k] = [h_woe, 0., 0., 0.]

    for k, v in top_sad.items():
        a = top_angry[k] if k in top_angry.keys() else 0.0
        h = top_happy[k] if k in top_happy.keys() else 0.0
        o = top_others[k] if k in top_others.keys() else 0.0
        s_woe = np.tanh((v/total_counts[k])/(a+h+o+0.001))

        if k in word_woe.keys(): word_woe[k][1] = s_woe
        else: word_woe[k] = [0.0, s_woe, 0., 0.]

    for k, v in top_angry.items():
        h = top_happy[k] if k in top_happy.keys() else 0.0
        s = top_sad[k] if k in top_sad.keys() else 0.0
        o = top_others[k] if k in top_others.keys() else 0.0
        a_woe = np.tanh((v/total_counts[k])/(h+s+o+0.001))

        if k in word_woe.keys(): word_woe[k][2] = a_woe
        else: word_woe[k] = [0.0, 0.0, a_woe, 0.]

    for k, v in top_others.items():
        a = top_angry[k] if k in top_angry.keys() else 0.0
        s = top_sad[k] if k in top_sad.keys() else 0.0
        h = top_happy[k] if k in top_happy.keys() else 0.0
        o_woe = np.tanh((v/total_counts[k])/(a+s+h+0.001))

        if k in word_woe.keys(): word_woe[k][3] = o_woe
        else: word_woe[k] = [0.0, 0.0, 0.0, o_woe]
            
    print('done')
    return word_woe

def get_class_wise_counters(long_convs, labels):
    from collections import Counter
    happy_counts = Counter()
    sad_counts = Counter()
    angry_counts = Counter()
    others_counts = Counter()
    total_counts = Counter()

    for lc in tqdm(range(len(long_convs))):
        turns = long_convs[lc].split(' <eos> ')
        for turn in turns:
            for word in turn.split(' '):
                if len(word) <= 2:
                    continue
                if(label2emotion[labels[lc]] == 'happy'): happy_counts[word] += 1
                elif(label2emotion[labels[lc]] == 'sad'): sad_counts[word] += 1
                elif(label2emotion[labels[lc]] == 'angry'): angry_counts[word] += 1
                else: others_counts[word] += 1
                total_counts[word] += 1
    
    return happy_counts, sad_counts, angry_counts, others_counts, total_counts


def get_woe_vector(data_path, mode='train'):
    WORD_WOE_FILE = 'word_woe.pickle'

    # get preprocessed data
    if mode == 'train':
        indices, conversations, labels = preprocessData(data_path, mode='train', convert=True)

        # compute woe_vectors
        happy_counts, sad_counts, angry_counts, others_counts, total_counts = \
            get_class_wise_counters(conversations, labels)
    
        n_top_words = 1000

        words_woe = get_woe_per_word(happy_counts, sad_counts
                        , angry_counts, others_counts, total_counts, n_top_words)

        with open(WORD_WOE_FILE, 'wb') as handle:
            pickle.dump(words_woe, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif mode == 'test':
        indices, conversations = preprocessData(data_path, mode='test', convert=True)
        with open(WORD_WOE_FILE, 'rb') as handle:
                words_woe = pickle.load(handle)

        
    # split long conv
    turn_arr = split_turns(conversations, 3)
    print("Broken conversations into turns.")

    turn1_list = turn_arr[:, 0].tolist()
    turn2_list = turn_arr[:, 1].tolist()
    turn3_list = turn_arr[:, 2].tolist()

    woe_arr = np.ndarray(shape=(len(conversations), 3, 4), dtype=float)
    for idx in tqdm(range(len(turn1_list))):
        words = turn1_list[idx].split(' ')
        for w in words:
            if w in words_woe.keys():
                woe_arr[idx][0][0] += words_woe[w][0]
                woe_arr[idx][0][1] += words_woe[w][1]
                woe_arr[idx][0][2] += words_woe[w][2]
                woe_arr[idx][0][3] += words_woe[w][3]

        words = turn2_list[idx].split(' ')
        for w in words:
            if w in words_woe.keys():
                woe_arr[idx][1][0] += words_woe[w][0]
                woe_arr[idx][1][1] += words_woe[w][1]
                woe_arr[idx][1][2] += words_woe[w][2]
                woe_arr[idx][1][3] += words_woe[w][3]

        words = turn3_list[idx].split(' ')
        for w in words:
            if w in words_woe.keys():
                woe_arr[idx][2][0] += words_woe[w][0]
                woe_arr[idx][2][1] += words_woe[w][1]
                woe_arr[idx][2][2] += words_woe[w][2]
                woe_arr[idx][2][3] += words_woe[w][3]
    woe_arr = np.round(woe_arr, 4)

    print('Collected WOE vectors.')
        
    return woe_arr

def transform_special_shape(turn1, turn2, turn3):
    g, t, p = [], [], []

    for i in range(len(turn1)):
        g.extend([turn1[i]])
        g.extend([turn2[i]])
        g.extend([turn3[i]])
        t.extend([turn1[i]])
        t.extend([""])
        t.extend([turn3[i]])
        p.extend([""])
        p.extend([turn2[i]])
        p.extend([""])
    return g, t, p

def array_permuted_indexing(a, n=3):
    m = a.shape[0]//n
    a3D = a.reshape(m, n, -1)
    return a3D[np.random.permutation(m)].reshape(-1,a3D.shape[-1])