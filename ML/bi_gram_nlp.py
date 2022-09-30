"""
to do
stopwords の除去
vectrizerで数値化
データ量が多すぎなので作品数上位20人か10人に変更
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import MeCab
import unidic
from nltk import ngrams

import sys
sys.path.append('..')
from preprocessings.preprocessing import clean_text, normalize, n_gram

row_data = pd.read_csv('../data/aozora_data2.csv')
row_data['author_num'] = row_data.groupby('author').ngroup()+1 
author_data = row_data[['author_num','author']]
text_data = row_data.text

df = pd.DataFrame()
df_add = pd.DataFrame()
l = []
bigram_l = []
for i in range(len(text_data[:3])):
    text_num_list = []
    num = author_data['author_num'][i]
    sentences = text_data[i].split('\n')
    for sentence in sentences:
        text = clean_text(sentence)
        sentence = normalize(text)
        
        tagger = MeCab.Tagger('-d ' + unidic.DICDIR)
        result = tagger.parse(sentence)
        node = tagger.parseToNode(sentence)
        nano_gram_count = {}
        # print(result)
        while node:
            word = node.surface
            hinshi = node.feature.split(",")[0]
            if hinshi in nano_gram_count.keys():
                freq = nano_gram_count[hinshi]
                nano_gram_count[hinshi] = freq + 1
            else:
                nano_gram_count[hinshi] = 1
            l.append(hinshi)
            # df_hinshi = pd.concat([df_add, df], axis=0)
            node = node.next
        # print(l)
        df_add = pd.DataFrame(l)
        # print(df_hinshi)
    #         # ずらしたものを入れる
        # df_add['next_hinshi'] = df_add['hinshi'].shift(1)
        df_add = df_add.rename(columns={0:'hinshi'})
        df_add['previous'] = df_add.shift(1)
        df_add = (df_add[1:].query("hinshi != 'BOS/EOS' & previous != 'BOS/EOS'"))

        amount = df_add.apply(lambda x: (x[0], x[1]), axis=1).value_counts(normalize=True)
        d = { 
            'author_num':num,
            'count':amount
        }
        bigram_l.append(d)