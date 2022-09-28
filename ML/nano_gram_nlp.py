# import文
import pandas as pd
import numpy as np

import MeCab
import unidic

import sys
sys.path.append('..')
from preprocessings.preprocessing import clean_text, normalize

"""
 **　todo

 *前処理
 後にfor文で１テキストずつ出力して全件整形
 dataframeで確認
sentenceにわけて一文字のやつを消す
 「上」「下」「中？」「一」「二」とかも消した方が良さそう？
 →消したら他に影響？
 数字は０に変更

 MeCabでいいけど
 辞書を変えると、固有名詞が結合されて、判別しにくい可能性
→どっちもやってみる
https://atmarkit.itmedia.co.jp/ait/articles/2102/05/news027_2.html

 *ML　分類　（svm、random forest, ）
 target(y):author
 expression(X):text, (title?)
 文脈、文の長さ、句読点などの書き方、位置で分類はできそう
 品詞情報でもできる

 *visualize
 shap とか、
 ダッシュボード
 →著者ごとに分類できたら、その著者を表す単語を著者ごとに分けて可視化
"""
# データの読み込み
row_data = pd.read_csv('../data/aozora_data.csv')
# print(row_data.groupby('author').size())
row_data['author_num'] = row_data.groupby('author').ngroup()+1 

author_data = row_data[['author_num','author']]
text_data = row_data.text
url = row_data.url
# print(author_data)

df = pd.DataFrame()
for i in range(len(text_data)):
    num = author_data['author_num'][i]
    text_num_list = []
    # print(text_data[i])
    # print('***'*50)
    sentences = text_data[i].split('\n')
    for sentence in sentences:
        text = clean_text(sentence)
        sentence = normalize(text)
        # print('---'*50)
        # print(sentence)
        tagger = MeCab.Tagger('-d ' + unidic.DICDIR)
        # tagger = MeCab.Tagger('-Owakati')
        result = tagger.parse(sentence)
        # print(result)
        node = tagger.parseToNode(sentence)
        # print(node)
        nano_gram_count = {}
        while node:
            word = node.surface
            hinshi = node.feature.split(",")[0]
            if hinshi in nano_gram_count.keys():
                freq = nano_gram_count[hinshi]
                nano_gram_count[hinshi] = freq + 1
            else:
                nano_gram_count[hinshi] = 1
            node = node.next
        del nano_gram_count['BOS/EOS']

        # total_freq = 0
        # freq_l = []
        text_dic = {}
        if nano_gram_count != {}:
            for key,value in nano_gram_count.items():
                # print(key+":"+str(value))
                freq = value/sum(nano_gram_count.values())
                # print(key+":"+str(freq))
                # total_freq += freq
            # print(total_freq)
            # print(int(value)/sum(int(value)))
            # print(nano_gram_count)
            # print(word +": "+ hinshi)
                text_dic[key]=freq
                # print(text_dic)
        else:
            continue
        text_num_list.append(text_dic)
                # d ={
                #     # 'author_num' : num,
                #     key : freq,
                #     # 'word_class_freq':freq_l,
                # }
    # print(text_num_list)
#                 text_num_list.append(d)
#         text_num_to_df = text_num_list
#     print(row_data.loc[i,'url'])
#     # print(text_num_list)
    df_add = pd.DataFrame(text_num_list)
    df_add['author_num'] = num
    df = pd.concat([df_add, df],axis=0)
    df =  df.fillna(0)
    # df = df.reset_index()
print(df)
# # print(text_num_list)

# print(df)
# """
# 品詞情報からまずは2値分類→多値分類
# ex.)芥川フラグ：１
#     その他：０

# 特定の品詞？だけまたはストップワードで重要な文章だけを取ってきたもので評価まで
# 名詞、動詞, etc...
# https://note.com/shimakaze_soft/n/nf02b0f8ab0f6 参考
# """





