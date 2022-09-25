# import文
from time import sleep
import configparser

from bs4 import BeautifulSoup
import re
import pandas as pd
import requests
import psycopg2 as pg

## 著作権がない作品を出力
# urlの取得
"""
あ行の作品からページごと、先頭文字毎に詳細ページへのURLを取得するための定義
保持するリストの初期化
"""
url = "https://www.aozora.gr.jp/index_pages/sakuhin_{alpha}{page}.html"
# 変数title_dic_listに空のリストを作成そこにタイトルとurlを格納
title_dic_list = []
html_url_list = []
text_url_list = []
d_list = [] 
alphas = ['a','i','u','e','o',
        'ka','ki','ku','ke','ko',
        'sa','si','su','se','so',
        'ta','ti','tu','te','to',
        'na','ni','nu','ne','no',
        'ha','hi','hu','he','ho',
        'ma','mi','mu','me','mo',
        'ya','yu','yo'
        'ra','ri','ru','re','ro'
        'wa','wo','nn','zz'
    ]


print('start')
# アクセスのためのページ番号とアルファベット
"""
ページ数はどのくらいあるのか確認していないので、念の為に５０ページまで
取得しようとさせる。
ページがなければリクエストでbreakし、文字の変更
"""
for alpha in alphas:
    for page in range(1,50):
        
        sleep(2)

        target_url = url.format(alpha=alpha, page=page)
    
    # # 適切にURLが取れてるか確認
    # target_urlへのアクセス結果を、変数rに格納
        r = requests.get(target_url)
        if r.status_code != 200:
            break 
        else:

        # BeautifulSoupに格納
            soup = BeautifulSoup(r.content, 'html.parser')
            # 不要なリンクを除く処理
            contents = soup.find_all(href=re.compile("../cards/"))
            # タイトルとリンクを取得
            for content in contents:
                title = content.text
                title_url = content
                title_url = title_url.get('href')
                title_url = 'https://www.aozora.gr.jp/' + title_url

                d = {
                    'title':title,
                    'title_url':title_url
                }
                # リストへ格納
                title_dic_list.append(d)  
                # ex.) https://www.aozora.gr.jp/../cards/001540/files/53897_50735.html


## 遷移後の画面でHTML(本文テキストがあるページ)のリンクを取得
print('well!')
print('get url')
"""
** to do →　１ページ毎に一つの作品に対してテキストまで取ってきて、SQLに保存するほうがもしかしたら
            エラーが少ないかもしれないので試してもよかった。。。？

         →　今回は全件のurlを取ってきて、詳細ページに入り、テキストのURL取得、テキスト抽出の方法でやってみた。
"""
for i in range(len(title_dic_list)):

    sleep(2)

    target_url = title_dic_list[i]['title_url']
   
    r = requests.get(target_url)
    if r.status_code != 200:

        """
        title_dic_list と数を合わせるために著作権がありページ構成が異なるerrorのURLとHTML形式でない
        タイトルも html_url_listに格納
        """

        print("error url",title_dic_list[i])
        d = {
                'html_url': "errorURL"
            }
        html_url_list.append(d)
        continue
        
    else:
        soup = BeautifulSoup(r.content, 'html.parser')
        contents = soup.find_all(href=re.compile("./files/"))
        #cardページに直接書籍が書いているページも除外
        if contents == []:
        
            """
            title_dic_list と数を合わせるために著作権があり、リンクが取得できないURLの
            タイトルも html_url_listに格納
            
            """

            print('no url',title_dic_list[i])
            d = {
                'html_url': 'noURL'
            }
            html_url_list.append(d)
            continue
        else:
            contents = contents[-1]
            
        # htmlのURLを取得
        html_url = contents.get('href')
        if html_url[-4:] != 'html':
            d = {
                'html_url':'noURL'
            }
            html_url_list.append(d)
        else:
            d = {
                    'html_url': html_url
                }
            # リストへ格納
            html_url_list.append(d)
            print(d)
        
"""
title_dic_list['title_url'][i][#..../cards/###] + html_url_list['html_url'][j][1:]
を取得したい
"""     
# テキストページへのURL作成
for i in range(len(html_url_list)):
    # 不要な文字の削除
    if html_url_list[i]['html_url'] != "noURL" or "errorURL":
        html_url = html_url_list[i]['html_url'][1:]
        title_url = title_dic_list[i]['title_url'][:40]

        text_url = title_url + html_url

        d = {
            'title' : title_dic_list[i]['title'],
            'text_url' : text_url
        }
        # テキストのタイトルとリンクを格納
        text_url_list.append(d)

print('well')

#　テキストを全件取得
print('get text')
for i in range(len(text_url_list)):

    sleep(2)

    target_url = text_url_list[i]['text_url']

    r = requests.get(target_url)
    if r.status_code != 200:
        print('not free', text_url_list[i])
        
    else:
        """
            カ行、ワ行で、atribute error その前のコード著作権があるものを抜けきれない文があったので、
            例外処理で一旦不要なものでも保存→今回は50件投稿がない人は除きたいので変更せず、SQLで除く
        """
        if text_url_list[i]['text_url'][-4:] != 'html':
            continue
            
        else:
            try:
                soup = BeautifulSoup(r.content, 'html.parser')
                title = soup.find('h1').text
                author = soup.find('h2').text
                text = soup.find('div', class_='main_text').text
            except AttributeError:
                print("AttributeError")
            
            d = {
                'get_title' : text_url_list[i]['title'],
                'url' : text_url_list[i]['text_url'],
                'title' : title,
                'author' : author,
                'text' : text
            }

            d_list.append(d)
            print(d)

print('finish')

# DataFrameに変換
text_data = pd.DataFrame(d_list)

print('send text to sql')
## SQLログイン情報 
conf = configparser.ConfigParser()
conf.read('config.ini')

path = conf['SQL']['path']
port = conf['SQL']['port']
dbname = conf['SQL']['dbname']
user = conf['SQL']['user']
password = conf['SQL']['password']

# SQLに保存
conText = "host={} port={} dbname={} user={} password={}"
conText = conText.format(path,port,dbname,user,password)

connection = pg.connect(conText)
cur = connection.cursor()

sql = "insert into text(get_title, url, title, author, text) values(%s, %s, %s, %s, %s);"
cur.executemany(sql, text_data.values)
connection.commit()

cur.close()

## postgreSQLテーブル
# CREATE table text (
# id serial unique Primary Key,
# get_title text,
# url text,
# title text,
# author text,
# text text
# )

print('all finish')
