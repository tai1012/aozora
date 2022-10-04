import pandas as pd 

data = pd.DataFrame()
data['品詞']=['名詞', '名詞', '名詞', '動詞', '動詞']
data['前の品詞'] = data['品詞'].shift(1)
print(data)
print(data.apply(lambda x: (x[0], x[1]), axis=1).value_counts())