import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../data/uni_gram_df2.csv')
# print(data.author_num)
print(data.corr())
# data = data.drop('Unnamed: 0', axis=1)
data = data[['名詞','動詞','形容詞','author_num']]

print(data)
# sns.pairplot(data, hue='author_num')
# X = data.drop('author_num', axis=1)
# y = data.author_num

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)

# # print(X_train.shape, X_test.shape)
# # print(y_train.shape, y_test.shape)
# # print(X_train.head())

# forest = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# print(forest.score(X_train, y_train))
# print(forest.score(y_test, y_pred))