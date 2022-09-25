# データの確認
import pandas as pd
import numpy as np

row_data = pd.read_csv('./data/aozora_data.csv')
print(row_data.shape)
data = row_data
# print(data.groupby(['author']).count())
# print(len(data.text))
text = np.array(data.text)
print(text)
