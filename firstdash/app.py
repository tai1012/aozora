from turtle import title
import cv2
import dash  
import dash_bootstrap_components as dbc
from dash import dcc, html
# import dash_shap_components as dashap
# import explainerdashboard as explain
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

import pandas as pd 
import numpy as np

import sys
sys.path.append('../../')

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

vec_df = pd.read_csv('../data/vec250.csv')
# wc_df = pd.read_csv('../data/aozora_data2.csv')

# 円グラフ
pie = vec_df.groupby('author').size()

fig_pie = go.Figure(
    data=[go.Pie(labels=list(pie.index),
                 values=pie.values,
                 marker=dict(colors=['Blues']))])

fig_pie.update_layout(
    width=350,
    height=300,
    # autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(x=0.0001,          # X座標
                y=0.000001,          # Y座標
                xanchor='left',  # X座標が凡例のどの部分を表すか
                yanchor='top',   # Y座標が凡例のどの部分を表すか
                orientation='h', # 凡例を横並びにする
            )
)
fig_pie.update_traces(
    textposition='inside', 
    textinfo='value+percent',
    direction='clockwise')

# 選択項目
author = {'選択なし':10,'坂口安吾':0,'太宰治':1,'宮本百合子':2,'宮沢賢治':3,
            '寺田寅彦':4,'小川未明':5,'寺田國士':6,'牧野信一':7,'芥川龍之介':8,'豊島与志雄':9}

sidebar = html.Div([
    dbc.Row(
        [
            html.P('Settings')
            ],
        style={"height":"5vh"}, 
        className='bg-primary text-white'
        ),
    dbc.Row(
        [
            html.P('Author Name',
                    style={'margin-top':'16px', 'margin-bottom':'4px'}
                ),
            dcc.Dropdown(id='author-picker', multi=False, value='all author',
                        options=[{'label': key, 'value': key}
                        for key, value  in author.items()],
                        style={'width': '320px','margin-top': '8px'},
                        className='font-weight-bold text-dark'
                        ),
            html.Button(id='button', n_clicks=0, children='apply',
                            style={'display': 'inline-block','width':'80px','margin-top':'16px','margin-left':'16px','margin-bottom':'4px'},
                            className='bg-dark text-white'),
            html.Hr()
            ],
        style={"height":"25vh"}, className='bg-secondary text-white'
        ),
    dbc.Row(
        [
            html.P('Target Variables',
                    style={'margin-top':'16px','margin-bottom':'4px'}
                ),
            html.Div('Total Title : 4308',
                    style={'margin-left':'24px'},
                    className='text_light'
                    ),
            dcc.Graph(figure=fig_pie)
            ],
        style={"height":"50vh"}, 
        className='bg-secondary text-white'
        ),
    dbc.Row([html.P('')],style={'height':'20vh'},className='bg-secondary text-white')
])

content = html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div([
                        html.H5('LightGBM Analytics'),
                        html.P(id='author-name1',
                            children='　　Selected Author : All Author'
                            )]) # 選択した著者名の表示 + author-name
                ],
                className='bg-info'
                )
        ],
        style={"height":"8vh"}
        ),
    dbc.Row([
        dbc.Col(
            [
                html.P('Word Cloud'),
                # html.Td(dcc.Graph(
                #         id='wc-img',
                #         figure = wc_fig,
                #         style={
                #                 'autosize':True,
                #                 # "height":300,
                #                 # 'width':800, 
                #                 'margin-left':'150px',
                #                 'margin-right':'100px',
                #                 'margin-bottom':'100px'
                #                 }
                # )),
                html.Td(html.Img(id = 'wc-img',
                                src="./static/wc10.png", # ワードクラウド変更、インタラクティブ
                                style={
                                    # 'autosize':True,
                                    "height":300,
                                    'width':800, 
                                    'margin-left':'150px',
                                    'margin-right':'100px',
                                    'margin-bottom':'100px'
                                    }))
            ],
            className='bg-white'
            )
        ],
        style={"height":"42vh"}
        ), 
    dbc.Row([
        dbc.Col([
            html.P('Top20 LightGBM Feature Importances'),
            html.Td(html.Img(id = 'feature-importance-img',
                            src="./static/importance.png", 
                            style={
                                "height":350,
                                'width':500, 
                                }))
        ],
        className='bg-light'
        ),
        dbc.Col([
            html.P(id='author-name2', 
                    children=f'Top20 Shap Feature Importances'), # タイトルの先頭に著者名
            html.Div(html.Img(id='class-shap-ex',
                            src="./static/shap10.png",  # shapのクラス変更をインタラクティブに
                            style={
                                "height":350,
                                'width':500, 
                                }))
        ],
        className='bg-light'
        )
    ],
    style={"height":"50vh"})
])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9)
                ],
            style={"height":"100vh",'autosize':True}
            ),
        ],
    fluid=True
    )



@app.callback(Output(component_id='author-name1', component_property='children'),
            Output(component_id='wc-img', component_property='src'),
            Output(component_id='author-name2', component_property='children'),
            Output(component_id='class-shap-ex', component_property='src'),
            Input(component_id='button', component_property='n_clicks'),
            State(component_id='author-picker', component_property='value'))

def update_name(n_clicks, author_pick):
    for k, v in author.items():
        if k == author_pick:
            author_num = v
    if n_clicks == 0:
       author_name1 = '　　Selected Author : Not selected'
       author_name2 = 'Top20 Shap Feature Importances'
    # author_num = [v  if ]
    # author_name1 = [k for k, v in author.items() if v == author_num]
    author_name1 =  '　　Selected Author : ' + author_pick
    wc_img = f'./static/wc{author_num}.png'
    # author_name2 = [k for k, v in author.items() if v == author_num]
    author_name2 = f'Top20 Shap Feature Importances :' + author_pick
    class_shap_ex = f'./static/shap{author_num}.png'

    return  author_name1, wc_img, author_name2, class_shap_ex

if __name__ == "__main__":
    app.run_server(debug=True, port=1234)

