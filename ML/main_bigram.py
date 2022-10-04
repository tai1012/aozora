"""
Bigram　でのデータ分析
** to do 
可視化、umap kmeans randomforest lightgbm svm, shap

情報量の削減
"""
# import文
import pandas as pd

# データの確認
row_data = pd.read_csv('../data/bigram_df2.csv')
row_data = row_data.drop('Unnamed: 0' , axis=1)
print(row_data.groupby('hinshi').size().sort_values(ascending=False).head(20))
# print(row_data)
# print(row_data[row_data['hinshi'].str.contains('名詞|助詞|動詞')])


