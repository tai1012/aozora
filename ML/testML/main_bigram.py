"""
Bigram　でのデータ分析
** to do 
可視化、umap kmeans randomforest lightgbm svm, shap

情報量の削減
"""
# import文
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns 
from mlxtend.plotting import plot_decision_regions


from sklearn.model_selection import train_test_split
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
import lightgbm as lgbm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import shap

# データの確認
row_data = pd.read_csv('../../data/bigram_df3.csv')
row_data = row_data.drop('Unnamed: 0' , axis=1)
name = pd.read_csv('../../data/aozora_data2.csv')
name['author_num'] = name.groupby('author').ngroup()+1

# 目的変数
name = name[['author', 'author_num']]

row_data = row_data[::-1].reset_index()
row_data = row_data.drop('index',axis=1)
name = name.rename(columns={'author':'author_name'})

data = pd.concat([name,row_data[::-1]], axis=1)
data = data.drop('author_num', axis=1)

# data.to_csv('../../data/bigram.csv')
# print(data["('動詞', '動詞')"])
# if data.groupby('author').mean() >= 0.01:

# print(data[data > 0.01])
# print(data.columns)

"""
可視化　pairplot 確認してみて、svmなどの次元を考慮したものを扱えば
うまく品詞情報だけでもできそうな感じ
"""
# data = data.iloc[:,0:11]
# viz_data = data.drop('author', axis=1)
# sns.pairplot(viz_data, hue='author_name')
# print(plt.show())

# data = data.iloc[:,1:20]
X = data.drop(['author', 'author_name'], axis=1)
y_name = data['author_name']
y = data['author']
# X = np.array(X)
# y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)


pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)
# print(X_train.head())

# umap
# mapper = umap.UMAP()
# embedding = mapper.fit_transform(X)
# embedding_x = embedding[:, 0]
# embedding_y = embedding[:, 1]

# for n in np.unique(y):
#      plt.scatter(embedding_x[y == n],
#                     embedding_y[y == n],
#                     label=n
#                )
# plt.grid()
# plt.legend()
# print(plt.show())
# # ランダムフォレスト　
# forest = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=42)
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# print(y_pred)
# print(forest.score(X_train, y_train))
# print(forest.score(X_test, y_test))

"""
結果　
     colab環境で、t-SNE,UMAP,KMeansによって可視化
     　→判断できなさそう
          →svmならもしかしたら。。。

     ランダムフォレスト：param　： n_estimators=10000, max_depth=20
     スコア　0.5978344934261408 
     特徴量　左から９個目までの品詞情報

     全特徴量使っても0.774168600154679 /  param : n_estimators=1000, max_depth=50

     いずれも過学習　testscore
"""



# # # print(data)




# # print(name.groupby(['author', 'author_num']).size())
# # print(row_data.groupby('author').size())
# # print(row_data)
# # print(row_data.groupby('hinshi').size().sort_values(ascending=False).head(20))
# # print(row_data)
# # print(row_data[row_data['hinshi'].str.contains('名詞|助詞|動詞')])

# # svm

svm = svm.SVC(C = 1000, gamma = 10, kernel = 'rbf', random_state=42)
svm.fit(X_train2, y_train)

pred_train = svm.predict(X_train2)
accuracy_train = accuracy_score(y_train, pred_train)
print(accuracy_train)

# pred_test = svm.predict(X_test)
accuracy_test = accuracy_score(X_test,y_test)
# print(pred_test, accuracy_test)


X_combined = X_test.values
y_combined = y_test.values

fig = plt.figure(figsize=(13,13))
plot_decision_regions(X_train2, y_train, clf=svm)
print(plt.show())

# X = data.drop(['author', 'author_name'], axis=1)
# y_name = data['author_name']
# y = data['author']

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)

# lgbm = lgbm.LGBMClassfier()
# lgbm.fit(X_train, y_train)

# pred = lgbm.predict(X_test)
# print(classification_report(pred, y_test, target_names=y_name))