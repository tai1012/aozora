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

author_data = row_data.author
text_data = row_data.text
url = row_data.url

# print(row_data.unique())
row_data['author_num'] = row_data.groupby('author').ngroup()+1
print(row_data['author_num'])
print(row_data.groupby(['author','author_num']).size())

row_data[['author','author_num']]