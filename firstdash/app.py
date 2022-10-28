import dash  
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_shap_components as dashap
import explainerdashboard as explain

import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd 
import numpy as np

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

df = pd.read_csv('../data/vec250.csv')

pie = df.groupby('author').size() / len(df)

fig_pie = go.Figure(
    data=[go.Pie(labels=list(pie.index),
                 values=pie.values,
                 marker=dict(colors=['Blues']))])

fig_pie.update_layout(
    width=350,
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(x=0.0001,          # ①：X座標
                y=0.000001,          # ①：Y座標
                xanchor='left',  # ②：X座標が凡例のどの部分を表すか
                yanchor='top',   # ②：Y座標が凡例のどの部分を表すか
                orientation='h', # ③：凡例を横並びにする
            )
)
fig_pie.update_traces(
    textposition='inside', 
    textinfo='percent',
    direction='clockwise')

sidebar = html.Div(
    [
        dbc.Row(
            [
                html.P('Settings')
                ],
            style={"height": "5vh"}, className='bg-primary text-white'
            ),
        dbc.Row(
            [
                html.P('Author Name')
                ],
            style={"height": "50vh"}, className='bg-secondary text-white'
            ),
        dbc.Row(
            [
                html.P('Target Variables'),
                dcc.Graph(figure=fig_pie)
                ],
            style={"height": "45vh"}, 
            className='bg-secondary text-white'
            )
        ]
    )

content = html.Div(
    [   dbc.Row(
            [
                dbc.Col(
                    [
                        html.P('Prediction')
                    ],
                    className='bg-info'
                    )
            ],
            style={"height": "8vh"}
            ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P('Word Cloud')
                    ],
                    className='bg-white'
                    )
            ],
            style={"height": "42vh"}
            ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P('something'),
                    ],
                    className='bg-light'
                    ),
                dbc.Col(
                    [
                        html.P('Feature Importance'),
                    ],
                    className='bg-light'
                    )
            ],
            style={"height": "50vh"})
        ]
    )

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9)
                ],
            style={"height": "100vh"}
            ),
        ],
    fluid=True
    )

if __name__ == "__main__":
    app.run_server(debug=True, port=1234)

