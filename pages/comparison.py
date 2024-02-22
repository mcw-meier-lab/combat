import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import (
    html, dcc, dash_table,
    callback, Input, Output, State,
)
from dash.exceptions import PreventUpdate

import base64
import io
import pandas as pd

dash.register_page(
    __name__,
    path='/comparison',
    title='Combat Comparisons',
    name='Combat Comparisons'
)

@callback(
    Output('compare-stored-data', 'data'),
    Input('compare-upload-data', 'contents'),
    State('compare-upload-data', 'filename'),
    State('compare-upload-data', 'last_modified'),
    prevent_initial_call=True
)
def update_upload(contents, filename, list_of_dates):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    # Read data
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),low_memory=False)
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
                             delimiter=' ')
        elif 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
                             delimiter='\t')
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    return df.to_json(date_format='iso', orient='split')

@callback(
    Output('compare-data-container', 'children'),
    Input('compare-stored-data', 'data'),
    prevent_initial_call=True
)
def output_from_store(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), 
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        page_size=5,
        id="compare-data-table"
    )], style={'width':'98%'},id="compare-data-container")
    
    return table

@callback(
    Output("compare-data-output","children"),
    Input("compare-dropdown-voi","value"),
    Input("compare-stored-data","data"),
)
def compare_data(selected_voi,stored_data):
    pass

layout = html.Div([
    html.P(),
    html.Div([
        dbc.Card(
            dbc.CardBody([
                html.H4("Upload Data", className="card-title"),
                html.P(
                    "Choose some text data to upload and analyze."
                    " Preferrably CSV data.",
                    className="card-text"
                ),
                dcc.Upload(dbc.Button('Upload File',color="primary",class_name="me-1"),id='compare-upload-data'
                ),
                html.Hr(),
                html.Div(dash_table.DataTable(id='compare-data-table'),id="compare-data-container"),
            ])
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4("Choose Combat versions to compare.",className="card-title"),
                dbc.Checklist(options=[
                    {"label":"Fortin's neuroCombat"},
                    {"label":"Pomponio's neuroHarmonize"},
                    {"label":"ENIGMA Combat"}
                ],
                id="compare-checklist"),
                html.P(
                    "Choose a variable of interest "
                    "(such as a biological factor) "
                    "that you would like to compare."
                ),
                dcc.Dropdown(id="compare-dropdown-voi",placeholder="Variable of interest"),
                html.Hr(),
                html.Div(id="compare-combat-output")
            ])
        )
    ]),
    dcc.Store(id="compare-stored-data",storage_type="session") 
])