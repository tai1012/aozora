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

import pandas as pd

import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import lightgbm as lgbm

import shap

import sys
sys.path.append('../')
from preprocessings.preprocessing import clean_text, normalize, set_stopwords, extract

# データフレームを綺麗に出力する関数
import IPython
def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)

row_data = pd.read_csv('../data/aozora_data2.csv')
row_data['author_num'] = row_data.groupby('author').ngroup()+1 
author = row_data[['author_num','author']]
text_data = row_data.text
title = row_data.title

text_l = []
for i in range(len(text_data)):
    # テキスト単位でのクリーニングと正規化
    text = clean_text(text_data[i])
    text = normalize(text)
    text = extract(text)
    text_l.append(text)
    # print(text)
title_l = []
for i in range(len(title)):
    title_l.append(title[i])

# モデルを生成
vectorizer = TfidfVectorizer(
    token_pattern=u'(?u)\\b\\w+\\b',
    max_features=250
    )

X = vectorizer.fit_transform(text_l)
values = X.toarray()
feature_names = vectorizer.get_feature_names()
df = pd.DataFrame(values, columns=feature_names)
df_add = pd.concat([author, title],axis=1)
df = pd.concat([df, df_add], axis=1)

data = df
y = data.author_num.values
X = data.drop(['author_num', 'author', 'title'], axis=1).values
feature_names = data[:-3].columns
feature_names = feature_names[:-3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

svm = SVC(C=120, gamma=0.0095)               # インスタンス生成
svm.fit(X_train, y_train) # SVM実行

# 精度検証用
y_pred = svm.predict(X_test) # テストデーテへの予測実行
print(svm.score(X_train, y_train))
print(metrics.accuracy_score(y_test, y_pred))
# print(y_pred, y_test)

shap.initjs()

explainer = shap.Explainer(model = svm.predict, 
                           masker = X_train,
                           feature_names = feature_names)
shap_values = explainer(X_test)

explainer2 = shap.Explainer(model = svm.predict, 
                            masker = X_train,
                            feature_names = feature_names,
                            max_evals = 501
                            )
shap_values2 = explainer2(X_test) 

