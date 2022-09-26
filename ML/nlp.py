from unittest import result
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
 「上」「下」「中？」「一」「二」とかも消した方が良さそう？
 →消したら他に影響
 数字は０に変更

 Mecabでいいけど
 辞書、固有名詞が結合されて、判別しにくい可能性
→どっちもやってみる
https://atmarkit.itmedia.co.jp/ait/articles/2102/05/news027_2.html

 *ML　分類　（svm、random forest, ）
 target(y):author
 expression(X):text, (title?)
 文脈、文の長さ、句読点などの書き方、位置で分類はできそう

 *visualize
 shap とか、
 ダッシュボード
 →著者ごとに分類できたら、その著者を表す単語を著者ごとに分けて可視化
"""
# データの読み込み
row_data = pd.read_csv('../data/aozora_data.csv')
# print(row_data.groupby('author').size())

author_data = row_data.author
text_data = row_data.text
url = row_data.url

# print(author_data.head())
# print(text_data.head())
# print(row_data.head())
text = np.array(text_data)
print(text[1])
# print(text_data.head())
# for i in range(len(text[0:5])):
print('***'*50)
text = clean_text(text[1])
text = normalize(text)

tagger = MeCab.Tagger()
result = tagger.parse(text)
print(result)
    # print(clean_text(text[i]))
    # print(normalize(text[i]))

    # print(row_data.loc[i,'url'])






# print(text_data[5]) #二つ目のテキストがみやすそうなので一旦実装


