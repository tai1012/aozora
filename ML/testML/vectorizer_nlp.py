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
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import scipy
from mlxtend.plotting import plot_decision_regions
from wordcloud import WordCloud
from PIL import Image

import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, f1_score
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import lightgbm as lgbm

import shap

import sys
sys.path.append('../../')
from preprocessings.preprocessing import clean_text, normalize, set_stopwords, extract

# データフレームを綺麗に出力する関数
import IPython
def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)

row_data = pd.read_csv('../../data/aozora_data2.csv')
row_data['author_num'] = row_data.groupby('author').ngroup()
author = row_data[['author_num','author']]
# text_data = row_data.query('author_num==9')['text']  #ワードクラウド用
text_data = row_data.text
title = row_data.title
# print(row_data)
# print(author)
# print(title)
text_l = []
for i in range(len(text_data)):
    # テキスト単位でのクリーニングと正規化
    text = clean_text(row_data.text[i])
    text = normalize(text)
    text = extract(text)
    text_l.append(text)
    # print(text)
title_l = []
for i in range(len(title)):
    title_l.append(title[i])

## ワードクラウド
"""
to do...
動的なワードクラウドの作成

※今回png使用(2022/11)
"""
img = Image.open('../../wordcloud/1817/Small/1817.jpg')
mask_img = np.array(img)
fpath = "/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
wordcloud = WordCloud(mask=mask_img,background_color="white",font_path=fpath,width=600,height=400,min_font_size=15,stopwords=set_stopwords())
wordcloud.generate(" ".join(text_l))
# wordcloud.to_file('../../wordcloud/wc_toyoshima.png')

# モデルを生成
vectorizer = TfidfVectorizer(
    token_pattern=u'(?u)\\b\\w+\\b',
    max_features=250,
    smooth_idf=False,
    analyzer='word',
    stop_words=set_stopwords()
    )




X = vectorizer.fit_transform(text_l)
# print(vectorizer.inverse_transform(X))
# y = author.author
# データフレームに表現
values = X.toarray()
feature_names = vectorizer.get_feature_names()
df = pd.DataFrame(values, columns=feature_names)
df_add = pd.concat([author, title],axis=1)
df = pd.concat([df, df_add], axis=1)
data = df
# print(data)
vectorizer.inverse_transform(X)
y = data.author_num
X = data.drop(['author_num', 'author', 'title'], axis=1)
feature_names = data[:-3].columns
feature_names = feature_names[:-3]
# print(df)
# print('to_csv')
# df.to_csv('../../data/vec250.csv')
"""
# # データ用意
#条件設定
# max_score = 0
# SearchMethod = 0
# SVC_grid = {SVC(): {"C": [10 ** i for i in range(-5, 6)],
#                     "kernel": ["linear", "rbf", "sigmoid"],
#                     "decision_function_shape": ["ovo", "ovr"],
#                     "random_state": [i for i in range(0, 101)]
#                    }}
# SVC_random = {SVC(): {"C": scipy.stats.uniform(0.00001, 1000),
#                     "kernel": ["linear", "rbf", "sigmoid"],
#                     "decision_function_shape": ["ovo", "ovr"],
#                     "random_state": scipy.stats.randint(0, 100)
#                      }}

#トレーニングデータ、テストデータの分離
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=0)
print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

#グリッドサーチ
for model, param in SVC_grid.items():
    clf = GridSearchCV(model, param)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

#ランダムサーチ
for model, param in SVC_random.items():
    clf =RandomizedSearchCV(model, param)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")

    if max_score < score:
        SearchMethod = 1
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

if SearchMethod == 0:
    print("サーチ方法:グリッドサーチ")
else:
    print("サーチ方法:ランダムサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))

#ハイパーパラメータを調整しない場合との比較
model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("")
print("デフォルトスコア:", score)
# SVM実行
# model = SVC()               # インスタンス生成
# model.fit(X_train, y_train) # SVM実行
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# forest = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=42)
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# print(y_pred)
# print(forest.score(X_train, y_train))
# print(forest.score(X_test, y_test))

# Feature Importance
# fti = forest.feature_importances_   

# print('Feature Importances:')
# for i, feat in enumerate(feature_names):
#     print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

# svm = SVC(C=120, gamma=0.0095)               # インスタンス生成
# svm.fit(X_train, y_train) # SVM実行

# pca = PCA(n_conponents=2)
# X_pca = pca.fit_transform(X_train)



# 予測実行
      # 精度検証用
# y_pred = svm.predict(X_test) # テストデーテへの予測実行
# print(svm.score(X_train, y_train))
# print(metrics.accuracy_score(y_test, y_pred))
# print(y_pred, y_test)

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
# shap
# shap.initjs()
# explainer = shap.Explainer(model=svm.predict,
#                             masker=X_train,
#                             feature_names=feature_names
#                         )
# shap_values = explainer(X_test)
# print(shap.plots.bar(shap_values))


# for C in [0.01, 1, 100]:
#     for gamma in [0.1, 1, 10]:
#         clf = SVC(C=C, gamma=gamma)
#         clf.fit(X_train, y_train)
#         train_score = clf.score(X_train, y_train)
#         print('train :', train_score)
#         ac_score = metrics.accuracy_score(y_test, clf.predict(X_test))
#         print(ac_score, "C = %s, gamma = %s" % (C, gamma))
# # labels = list(author.author.unique())

# # eval_dict = classification_report(y_test, y_pred, output_dict=True, target_names=labels)
# # df = pd.DataFrame(eval_dict)  # DataFrameとして表示
# # print(df)

# # pd.options.display.precision = 4

# # plt.figure(figsize=(15, 15))
# # cm = confusion_matrix(y_test, y_pred)
# # cm = pd.DataFrame(cm, index=labels, columns=labels)
# # sns.heatmap(cm, annot=True, square=True, cmap='Blues')
# # plt.yticks(rotation=0)
# # plt.xlabel("Pre", fontsize=13, rotation=0)
# # plt.ylabel("GT", fontsize=13)
# # print(plt.show())


"""

# lightGBM
均衡データ
デフォルト、パラメータの設定
"""
# df_lgb = lgbm.LGBMClassifier()
# df_lgb.fit(X_train, y_train)
# y_pred = df_lgb.predict(X_test)
# y_pred_prob = df_lgb.predict_proba(X_test)
# y_pred_max = np.argmax(y_pred_prob, axis=1) +1
# df_pred = pd.DataFrame({'target':y_test,'target_pred':y_pred})
# print(df_pred)

# # 真値と予測確率の表示
# df_pred_prob = pd.DataFrame({'y':y_test, 'target_prob' : y_pred_max})
# print(df_pred_prob)

# train_score = df_lgb.score(X_train, y_train)
# print("train:", train_score)
# acc = df_lgb.score(X_test, y_test)
# print('Acc :', acc)

# パラメータあり
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)

clf = lgbm.LGBMClassifier(
    objective='multiclass',
    n_estimators=2000,
    early_stopping_rounds=50, 
    num_class=10,
    # num_leaves=10, #
    max_depth=2, #
    learning_rate=0.01,
    importance_type='gain',
    # random_state=42
)
clf.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100)
# plot_confusion_matrix(lgbm,X_test,y_test)
# plt.show()

y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
y_pred_max = np.argmax(y_pred_prob, axis=1)
# df_pred = pd.DataFrame({'target':y_test,'target_pred':y_pred})
# print(df_pred)

# # 真値と予測確率の表示
df_pred_prob = pd.DataFrame({'y':y_test, 'target_prob' : y_pred_max})
print(df_pred_prob)

train_score = clf.score(X_train, y_train)
print("train:", train_score)
acc = clf.score(X_test, y_test)
print('Acc :', acc)
print(clf.get_params())

"""
# 250words nocharactor
# 評価　default train : 0.9953565505804312
#               test : 0.868522815158546
# train: 0.9957345971563981
# Acc : 0.8406805877803558

250words 一文字以上の名詞

train: 0.9440758293838862
Acc : 0.8267594740912606

"""

feature = pd.Series(clf.feature_importances_,index=feature_names)
print(feature)

print(lgbm.plot_importance(clf, max_num_features=20))
plt.savefig('../../data/importance.png')
# print(plt.show())
plt.close()

shap.initjs()

explainer = shap.TreeExplainer(model = clf, 
                           data = X_train,
                           feature_names = feature_names)
shap_values = explainer.shap_values(X_train)
print(shap_values)

for i in range(10):
    if i == 0:
        fig = plt.gcf()
        print('All Author')
        plt.title("All Author")
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        fig.set_size_inches(15, 10, forward=True)
        fig.savefig("../../firstdash/static/all_shap.png")
        plt.close()
    fig = plt.gcf()
    print('Class ', i)
    plt.title(f'Class{i}')
    shap.summary_plot(shap_values[i], X_train, plot_type="bar", show=False)
    fig.set_size_inches(15, 10, forward=True)
    fig.savefig(f"../../firstdash/static/shap{i}.png")
    plt.close()