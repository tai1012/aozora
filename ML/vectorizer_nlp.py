"""
to do
stopwords の除去
vectrizerで数値化
データ量が多すぎなので作品数上位20人か10人に変更

辞書はデフォルトの方が精度が良さそう

※品詞情報
名詞、動詞に絞ってTF-IDFベクトル化　cos類似度
助詞は不要（ストップワードに多いイメージ）　

※可視化
pairplot等

※モデル　
ランダムフォレスト、サポートベクトルマシン、lightGBM

※可視化
plotly, tableauみたいに動的に表示が変わるやつ
ダッシュボードの作成
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import MeCab
import unidic
# from nltk import ngrams

import sys
sys.path.append('../')
from preprocessings.preprocessing import clean_text, normalize, set_stopwords, extract

row_data = pd.read_csv('../data/aozora_data2.csv')
row_data['author_num'] = row_data.groupby('author').ngroup()+1 
author = row_data[['author_num','author']]
text_data = row_data.text
title = row_data.title
# print(row_data)
# print(author)
# print(title)
text_l = []
for i in range(len(text_data)):
    # テキスト単位でのクリーニングと正規化
    text = clean_text(text_data[i])
    text = normalize(text)
    text = extract(text)
    text_l.append(text)
    print(text)
title_l = []
for i in range(len(title)):
    title_l.append(title[i])

# モデルを生成
vectorizer = TfidfVectorizer(
    # token_pattern=u'(?u)\\b\\w+\\b',
    max_features=250
    )
X = vectorizer.fit_transform(text_l)
y = author.author_num
# データフレームに表現
values = X.toarray()
feature_names = vectorizer.get_feature_names()
df = pd.DataFrame(values, columns=feature_names)
df_add = pd.concat([author, title],axis=1)
df = pd.concat([df, df_add], axis=1)
# print(df)
# df.to_csv('../data/vec_250words_nochar.csv')
# # データ用意
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# SVM実行
from sklearn.svm import SVC # SVM用
model = SVC()               # インスタンス生成
model.fit(X_train, y_train) # SVM実行

# 予測実行
from sklearn import metrics      # 精度検証用
predicted = model.predict(X_test) # テストデーテへの予測実行
print(model.score(X_train, y_train))
print(metrics.accuracy_score(y_test, predicted))
print(predicted, y_test)

"""
svm評価      train_score : 0.9996683250414594 過学習っぽい
            test_score : 0.8607888631090487 名詞の単語数の絞り込みなし

            50000語まで
            train : 0.9996683250414594
            test : 0.8646558391337974

            10000語まで
            train : 0.9993366500829187
            test : 0.8839907192575406

            5000語
            train : 0.9993366500829187
            test : 0.8924980665119876

            500語
            train : 0.9933665008291874
            test : 0.8924980665119876

            # 100
            # train : 0.9459369817578773
            # test : 0.8306264501160093

            # 250
            # train : 0.9824212271973466
            # test : 0.8669760247486465

            400
            train : 0.9903814262023217
            test : 0.8808971384377416

            300
            train : 0.9864013266998342
            test : 0.880123743232792

        一単語しかないものを除外
            250
            train : 0.9611940298507463
            test : 0.8360402165506574

            100
            train : 0.9177446102819237
            test : 0.8035576179427688

            500
            train : 0.9827529021558872
            test :　0.8375870069605569

"""
