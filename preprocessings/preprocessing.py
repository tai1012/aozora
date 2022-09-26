# 前処理用関数の作成
# import文
import re
import unicodedata

import nltk
from nltk.corpus import wordnet

"""
text = "①1１"
>>> unicodedata.normalize("NFKC",text)
'111'
NFKCはバイト列を短縮できるので比較して軽い

unicode 正規化：https://qiita.com/fury00812/items/b98a7f9428d1395fc230
"""
# 文章のクリーニング
def clean_text(text):
    replaced_text = text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    # replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", ' ', replaced_text) # 全角記号の除去 !,/,:,@,[],``,{},., 
    replaced_text = re.sub(
        r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    replaced_text = re.sub(r'\r\n', '', replaced_text)  # 全角空白の除去
    replaced_text = re.sub(r'\r', '', replaced_text)  # 全角空白の除去
    replaced_text = re.sub(r'\n', ' ', replaced_text)  # 全角空白の除去
    return replaced_text

# 正規化
def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text
    
def lower_text(text):
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def normalize_number(text):
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text
    
def lemmatize_term(term, pos=None):
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)



