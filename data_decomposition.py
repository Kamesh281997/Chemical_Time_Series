from main import connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


def decomposition(df): 
    # df = pd.read_csv(path)     
    df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
    df.set_index('hourly_timestamp', inplace=True)
      
    columns_to_decompose = ['tbl_speed', 'fom', 'main_comp', 'tbl_fill', 'srel', 'produced', 'waste', 'cyl_main', 'stiffness', 'ejection']
    app = dash.Dash(__name__)

    # Custom styling
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'height': '100vh'},
        children=[
            html.H1("Time Series Decomposition Dashboard", style={'text-align': 'center'}),

            html.Div([
                dcc.Dropdown(
                    id='batch-dropdown',
                    options=[{'label': str(batch), 'value': batch} for batch in df['batch'].unique()],
                    value=df['batch'].min(),
                    multi=False,
                    style={'width': '45%', 'margin-right': '5%', 'display': 'inline-block', 'backgroundColor': '#FFFFFF', 'color': '#111111'}
                ),

                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in columns_to_decompose],
                    value=columns_to_decompose[0],
                    multi=False,
                    style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#FFFFFF', 'color': '#111111'}
                ),
            ], style={'margin-bottom': '20px'}),

            dcc.Dropdown(
                id='component-dropdown',
                options=[
                    {'label': 'Observed', 'value': 'observed'},
                    {'label': 'Residual', 'value': 'residual'},
                    {'label': 'Trend', 'value': 'trend'},
                    {'label': 'Seasonal', 'value': 'seasonal'}
                ],
                value='observed',
                multi=False,
                style={'width': '50%', 'margin-bottom': '20px', 'backgroundColor': '#FFFFFF', 'color': '#111111'}
            ),

            html.Div(id='decomposition-plots'),
        ]
    )

    @app.callback(
        Output('decomposition-plots', 'children'),
        [Input('batch-dropdown', 'value'),
         Input('feature-dropdown', 'value'),
         Input('component-dropdown', 'value')]
    )
    def update_plots(selected_batch, selected_feature, selected_component):
        try:
            batch_data = df[df['batch'] == selected_batch]
            result = seasonal_decompose(batch_data[selected_feature], model='additive', period=2)

            if selected_component == 'observed':
                component_fig = px.line(x=batch_data.index, y=result.observed, title=f'Observed ({selected_feature})')
            elif selected_component == 'residual':
                component_fig = px.line(x=batch_data.index, y=result.resid, title=f'Residuals ({selected_feature})')
            elif selected_component == 'trend':
                component_fig = px.line(x=batch_data.index, y=result.trend, title=f'Trend ({selected_feature})')
            elif selected_component == 'seasonal':
                component_fig = px.line(x=batch_data.index, y=result.seasonal, title=f'Seasonal ({selected_feature})')

            return [dcc.Graph(figure=component_fig)]
        except Exception as e:
            return f"Error: {str(e)}"
    app.run_server(debug=True)
    
    
    



    

