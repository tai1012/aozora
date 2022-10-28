# 前処理用関数の作成
# import文
import re
import unicodedata
import urllib

import MeCab
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
    replaced_text = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', replaced_text) # 全角記号の除去 !,/,:,@,[],``,{},., 
    replaced_text = re.sub(
        r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', '', replaced_text)  # 全角空白の除去
    # replaced_text = re.sub(r'. … -', '', replaced_text)  # .,…,-の除去
    replaced_text = re.sub(r'\r', '', replaced_text)  # 先頭の空白の除去
    replaced_text = re.sub(r'\n', '', replaced_text)  # 改行の除去
    replaced_text = re.sub(r'――', '', replaced_text)  # .,…,-の除去
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

def wakachi(text):

    mecab = MeCab.Tagger("-Owakati")

    return mecab.parse(text).strip().split(" ")


def n_gram(target, n):
  # 基準を1文字(単語)ずつ ずらしながらn文字分抜き出す
    return [ target[idx:idx + n] for idx in range(len(target) - n + 1)]

# ストップワードの設定を行う関数を定義。今回はローカルのtxtファイルから設定できるようにした。
def set_stopwords():
    """
    Get stopwords from input document.
    """
    # Defined by SlpothLib
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    
    stopwords_list = []
    
    # add stop_word from text file
    f = open('more_stopword.txt')
    txt_file = f.readlines()
    f.close()
    more_stopwords = [line.strip() for line in txt_file]
    more_stopwords = [ss for ss in more_stopwords if not ss==u'']
    stopwords_list += more_stopwords

    # Merge and drop duplication
    stopwords_list += slothlib_stopwords
    stopwords_list = set(stopwords_list)

    return stopwords_list

# 品詞抽出、分かち書き
tagger = MeCab.Tagger("")
tagger.parse("")

def extract(text):
    words = []

    # 単語の特徴リストを生成
    node = tagger.parseToNode(text)

    while node:
        # 品詞情報(node.feature)が名詞ならば
        if node.feature.split(",")[0] == u"名詞" and node.feature.split(",")[2] == u"一般":
            # 単語(node.surface)をwordsに追加
            words.append(node.surface)
        node = node.next

    # 半角スペース区切りで文字列を結合
    text_result = ' '.join(words)
    return text_result