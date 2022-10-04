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
sys.path.append('../../')
from preprocessings.preprocessing import clean_text, normalize, n_gram

row_data = pd.read_csv('../../data/aozora_data2.csv')
row_data['author_num'] = row_data.groupby('author').ngroup()+1 
author_data = row_data[['author_num','author']]
text_data = row_data.text

df = pd.DataFrame()
# df_add = pd.DataFrame()

bigram_l = []
# for i in range(len(text_data)):
for i in range(10):
    l = []
    text_list = []
    num = author_data['author_num'][i]
    # sentences = text_data[i].split('\n')
    # for sentence in sentences;
    text = clean_text(text_data[i])
    text = normalize(text)
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    nano_gram_count={}
    while node:
        word = node.surface
        hinshi = node.feature.split(",")[0]
        if hinshi in nano_gram_count.keys():
            freq = nano_gram_count[hinshi]
            nano_gram_count[hinshi] = freq + 1
        else:
            nano_gram_count[hinshi] = 1
        l.append(hinshi)
        # print(hinshi)
        # df_hinshi = pd.concat([df_add, df], axis=0)
        node = node.next
       
    df_add = pd.DataFrame(l)
    # print(df_hinshi)
    # ずらしたものを入れる
    df_add = df_add.rename(columns={0:'hinshi'})
    df_add['previous'] = df_add.shift(1)
    df_add = df_add[1:].query("hinshi != 'BOS/EOS' & previous != 'BOS/EOS'")
    # df_add = df_add.query('hinshi == 助詞  | previous == 名詞 & hinshi == 名詞')
    # df_add['author_num'] = num
    # for index, value in df_add.value_counts().iteritems():
    #     if value >= 100:    
    #         df['hinshi'] = index[0]
    #         df['previous'] = index[1]
    #         df['author'] = index[2]
    #         df['']
    #         print(index[0], index[1], index[2] ,value)
    amount = df_add.apply(lambda x: (x[0], x[1]), axis=1).value_counts(normalize=True)
    amount = dict(amount)
    df_bi = pd.DataFrame()
    df_bi['author'] = num
    for key, value in amount.items():
        df_bi[key] = pd.Series(value)
        # print(value)
        
    df = pd.concat([df_bi, df], axis=0)
    df = df.fillna(0)
print(df)
# df.to_csv('../../data/test_df2.csv')
#     df_bi = amount.rename_axis('hinshi').reset_index(name='counts')
#     df_bi['author'] = num
#     df = pd.concat([df_bi, df],axis=0)
# #     df_d = dict(df_bi)
# #     bigram_l.append(df_d)
# #     print(bigram_l)
# # print(len(bigram_l))
# # df = pd.DataFrame(bigram_l)
# df = df.dropna(0)
# df = df.reset_index()
# df = df.drop('index', axis=1)

# print(df)
# df.to_csv('../../data/bigram_df.csv')
#     # amount = amount.to_dict()
#     # df_add['count'] = df_add.apply(lambda x: (x[0], x[1]), axis=1).map(amount)
#     # print(df_add)
#     # bi_l = df_add
#     # print(amount)

# #         for sentence in sentences:
# #         text = clean_text(sentence)
# #         sentence = normalize(text)
# #         tagger = MeCab.Tagger('-Owakati')
# #         result = tagger.parse(sentence)

# #         node = tagger.parseToNode(sentence)
# #         nano_gram_count = {}
# #         while node:
# #             word = node.surface
# #             hinshi = node.feature.split(",")[0]
# #             if hinshi in nano_gram_count.keys():
# #                 freq = nano_gram_count[hinshi]
# #                 nano_gram_count[hinshi] = freq + 1
# #             else:
# #                 nano_gram_count[hinshi] = 1
# #             l.append(hinshi)
# #             # df_hinshi = pd.concat([df_add, df], axis=0)
# #             node = node.next
    
# #     df_add = pd.DataFrame(l)
# #     # print(df_hinshi)
# #     ## ずらしたものを入れる
# #     # df_add['next_hinshi'] = df_add['hinshi'].shift(1)
# #     df_add = df_add.rename(columns={0:'hinshi'})
# #     df_add['previous'] = df_add.shift(1)
# #     df_add = (df_add[1:].query("hinshi != 'BOS/EOS' & previous != 'BOS/EOS'"))
# #     df_add['author_num'] = num
# #     amount = df_add.apply(lambda x: (x[0], x[1]), axis=1).value_counts(normalize=True)
# #     amount = amount.to_dict()
# #     df_add['count'] = df_add.apply(lambda x: (x[0], x[1]), axis=1).map(amount)
# #     bi_l = df_add
# #     print(bi_l)
# # #     bigram_l.append(bi_l)
# # # df = pd.DataFrame(bigram_l)
# # # print(df)
# #     # print(df_add)
# #     # print(bi_d)
# # # print(df_add)
# # # print(df_add)