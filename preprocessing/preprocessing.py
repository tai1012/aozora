import pandas as pd
import numpy as np

row_data = pd.read_csv('./data/aozora_data.csv')
text_data = np.array(row_data.text)
