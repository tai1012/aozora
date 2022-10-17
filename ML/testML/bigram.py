"""
Bigramへのデータ加工

データ量が多すぎるので作品数上位20人か10人に変更
データとしては単語Bigramにし、品詞情報のみで相対頻度を数値として捉える
著者は番号に一旦置き換える
"""
# import文
import pandas as pd

import MeCab
import unidic

import sys
sys.path.append('..')
from preprocessings.preprocessing import clean_text, normalize

# データ読み込み
row_data = pd.read_csv('../data/aozora_data2.csv')

# 著者名をもとに番号に変更
row_data['author_num'] = row_data.groupby('author').ngroup()+1 
author_data = row_data[['author_num','author']]
text_data = row_data.text

# 定義しておきたいデータフレームとリストの作成
df = pd.DataFrame()
bigram_l = []

"""
1つの作品をすべて品詞情報にし、一文を単語バイグラムに変更
1作品に含まれる品詞情報をカウントする
改行のところで、BOS/EOSという情報になるのでカウントした後に削除、
1文毎の単語バイグラムの完成

"""
# 全テキストを1つずつ分ける
for i in range(len(text_data)):
    # 各毎テキスト毎に捨てたいリストの作成
    l = []
    text_list = []

    # 著者番号
    num = author_data['author_num'][i]
    
    # テキスト単位でのクリーニングと正規化
    text = clean_text(text_data[i])
    text = normalize(text)

    # MeCabに渡す
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    # 単語単位で一旦カウントするために辞書を作成
    nano_gram_count={}

    # 各単語ごとの情報を抽出
    while node:
        hinshi = node.feature.split(",")[0]

        # 品詞のカウント
        if hinshi in nano_gram_count.keys():
            freq = nano_gram_count[hinshi]
            nano_gram_count[hinshi] = freq + 1
        else:
            nano_gram_count[hinshi] = 1

        l.append(hinshi)
        
        # 次の単語に移動
        node = node.next

    # 品詞のリストをデータフレームに変更
    df_add = pd.DataFrame(l)

    # 一つずらした品詞情報を追加
    df_add = df_add.rename(columns={0:'hinshi'})
    df_add['previous'] = df_add.shift(1)

    # データフレーム内の最初の単語はNoneで文章の区切りの情報も除く
    df_add = df_add[1:].query("hinshi != 'BOS/EOS' & previous != 'BOS/EOS'")
    
    # 事前に不要なものは抜けているので、単語バイグラムの相対頻度を各テキスト各文章毎を抽出
    amount = df_add.apply(lambda x: (x[0], x[1]), axis=1).value_counts(normalize=True)
    df_bi = amount.rename_axis('hinshi').reset_index(name='freq')
    df_freq = pd.DataFrame()
    df_freq['author'] = pd.Series(num)
    # 品詞情報を列に持ちたいので変更
    for key, value in amount.items():
        df_freq[key] = pd.Series(value)
    df = pd.concat([df_freq, df],axis=0)
    
# テキストごとにない品詞の組み合わせがあるので０に変更
df = df.fillna(0)

df.to_csv('../data/bigram_df3.csv')
    