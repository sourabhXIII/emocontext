# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:11:51 2019

@author: sourabhxiii
"""

import os
os.chdir(r'C:\__MyComputer\OneDrive - Teradata\Drive_SM\Course\emo_context')

from collections import Counter
import numpy as np
import pandas as pd


df = pd.read_csv('data/train.txt', sep='\t')
labels = list(df['label'])
convs = df['turn1'].astype('str') + df['turn2'].astype('str') + df['turn3'].astype('str')
convs = convs.tolist()


        
happy_counts = Counter()
sad_counts = Counter()
angry_counts = Counter()
others_counts = Counter()
total_counts = Counter()


import emoji
import re

# replace emojis with text
turn1 = list(df['turn1'])
turn1_lines = []
for i in range(len(turn1)):
    conv = re.sub(':', '', re.sub('::', '_', emoji.demojize(turn1[i]))).split('_')
    # conv is a list now
    turn1_lines.append(" ".join(conv))

    
for i in range(len(convs)):
    conv = re.sub(':', '', re.sub('::', '_', emoji.demojize(convs[i]))).split('_')
    # conv is a list now
    for phrase in conv:
        for word in phrase.split(' '):
            if(labels[i] == 'happy'): happy_counts[word] += 1
            elif(labels[i] == 'sad'): sad_counts[word] += 1
            elif(labels[i] == 'angry'): angry_counts[word] += 1
            else: others_counts[word] += 1
            total_counts[word] += 1
        
            
pos_neg_ratios = Counter()

for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = happy_counts[term] / float(sad_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

for word,ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))
        
        
        
def array_permuted_indexing(a, n=4):
    m = a.shape[0]//n
    a3D = a.reshape(m, n, -1)
    return a3D[np.random.permutation(m)].reshape(-1,a3D.shape[-1])