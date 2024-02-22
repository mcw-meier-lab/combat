import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_extensions.enrich import (
    callback, Output, Input, State,
    dcc, html, dash_table, DashLogger, ctx
)
from dash.exceptions import PreventUpdate

import io
import json
import base64
import pandas as pd
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from formulaic import Formula

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import dash_bio
from sklearn.decomposition import PCA

from config import impute_data

dash.register_page(
    __name__,
    path='/longitudinal',
    title='Longitudinal Combat',
    name='Longitudinal Combat'
)

@callback(
    Output('stored-long-data', 'data'),
    Input('upload-long-data', 'contents'),
    State('upload-long-data', 'filename'),
    State('upload-long-data', 'last_modified'),
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
    Output('long-data-container', 'children'),
    Input('stored-long-data', 'data'),
    prevent_initial_call=True
)
def output_from_store(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), 
        columns=[{"name": i, "id": i, "selectable":True} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        editable=True,
        column_selectable="multi",
        selected_columns=[],
        page_size=10,
        id="long-data-table"
    )], style={'width':'98%'},id="long-data-container")
    
    return table

@callback(
    Output('dropdown-long-voi','options'),
    Output("dropdown-long-voi","value"),
    Input('dropdown-long-batch','value'),
    Input('dropdown-long-idvar','value'),
    Input('dropdown-long-timevar','value'),
    Input('long-data-table','selected_columns'),
    Input('stored-long-data','data'),
)
def set_long_voi(selected_batch,selected_idvar,selected_timevar,selected_cols,stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    voi_options = [c for c in df.columns if c != selected_batch and \
                   c != selected_idvar and c != selected_timevar]
    voi_value = []
    if selected_cols:
        voi_value = [c for c in selected_cols if c!= selected_batch and \
                     c != selected_idvar and c != selected_timevar]
        voi_options = [c for c in selected_cols if c != selected_batch and \
                       c != selected_idvar and c != selected_timevar]


    return voi_options, voi_value

@callback(
    Output('dropdown-long-batch','options'),
    Input('dropdown-long-idvar','value'),
    Input('dropdown-long-timevar','value'),
    Input('stored-long-data','data'),
)
def set_long_batch(selected_idvar,selected_timevar,stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    batch_options = [c for c in df.columns if c != selected_idvar and \
                     c != selected_timevar]

    return batch_options

@callback(
    Output('dropdown-long-idvar','options'),
    Input('stored-long-data','data')
)
def set_long_idvar(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data),orient="split")
    idvar_options = [c for c in df.columns]

    return idvar_options

@callback(
    Output('dropdown-long-timevar','options'),
    Input('dropdown-long-idvar','value'),
    Input('stored-long-data','data')
)
def set_long_timevar(selected_idvar,stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data),orient="split")
    timevar_options = [c for c in df.columns if c != selected_idvar]

    return timevar_options

@callback(
    Output('long-combat-model-output','children'),
    Input('stored-long-data','data'),
    State('long-combat-model','value'),
    Input('long-combat-model-submit','n_clicks'),
    prevent_initial_call=True
)
def get_long_model_matrix(stored_data,combat_model,n_clicks):
    if stored_data is None or combat_model is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    X = Formula(combat_model).get_model_matrix(df)
    table = html.Div([
    dash_table.DataTable(
        X.to_dict('records'), [{"name": i, "id": i} for i in X.columns],
        style_table={'overflowX': 'scroll'},
        page_size=5,
    )], style={'width':'98%'}) 

    return table

@callback(
        Output("download-long-combat-csv","data"),
        Input("btn-long-download","n_clicks"),
        Input("stored-long-combat","data"),
        prevent_initial_call=True
)
def download_long_combat_table(n_clicks,stored_combat):
    if stored_combat is None or n_clicks == 0:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_combat), orient='split') 
    return dcc.send_data_frame(df.to_csv,"longitudinal_combat_output.csv")

@callback(
    Output("long-combat-run-output","children"),
    Output("long-txt","children"),
    Input("stored-long-combat","data"),
    Input('long-combat-run-submit','n_clicks'),
    Input("long-combat-stdout","data"),
    prevent_initial_call=True,
    log=True
)
def update_long_combat_table(stored_combat,n_clicks,combat_stdout,dash_logger: DashLogger):
    if stored_combat is None:
        raise PreventUpdate
    if ctx.triggered_id != "long-combat-run-submit":
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_combat), orient='split')
    output = "".join(json.loads(combat_stdout)["Output"])
    dash_logger.info(message=output,title=f"Longitudinal Combat finished!",style={"overflow":"scroll"})

    button = html.Div([
        dbc.Button("Download",id="btn-long-download",color="primary",n_clicks=0),
        dcc.Download(id="download-long-combat-csv")
    ])
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        page_size=20,
    )], style={'width':'98%'})
    
    return [table,button]


@callback(
    Output("stored-long-combat","data"),
    Output("long-combat-stdout","data"),
    Input('dropdown-long-idvar','value'),
    Input('dropdown-long-timevar','value'),
    Input("dropdown-long-batch","value"),
    [Input("dropdown-long-voi","value")],
    State('long-combat-model','value'),
    State('long-combat-ranef','value'),
    Input('long-combat-run-submit','n_clicks'),
    Input("stored-long-data","data"),
    prevent_initial_call=True,
    log=True
)
def run_long_combat(selected_idvar,selected_timevar,selected_batch,selected_voi,combat_model,combat_ranef,n_clicks,stored_data,dash_logger=DashLogger):
    if stored_data is None or combat_model is None or selected_batch is None \
        or selected_voi is None or combat_ranef is None \
            or selected_idvar is None or selected_timevar is None:
        raise PreventUpdate
    if ctx.triggered_id != "long-combat-run-submit":
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    columns = [c for c in selected_voi]

    voi_data = df[columns].apply(pd.to_numeric)
    voi_data[selected_batch] = df[selected_batch]

    # check for missing
    missing_data = False
    for col in voi_data.columns:
        if any(pd.isnull(voi_data[col])) and not all(pd.isnull(voi_data[col])):
            missing_data = True

    buf = []

    if missing_data:
        voi_data = impute_data(voi_data)

    with (ro.default_converter + pandas2ri.converter).context():
        r_dataframe = ro.conversion.get_conversion().py2rpy(df)
        features = ro.StrVector(selected_voi)
        longCombat = importr('longCombat')
        consolewrite_print_backup = rpy2.rinterface_lib.callbacks.consolewrite_print
        rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: buf.append(x)
        full_combat = longCombat.longCombat(
            idvar=selected_idvar,
            timevar=selected_timevar,
            batchvar=selected_batch,
            features=features,
            formula=combat_model,
            ranef=combat_ranef,
            data=r_dataframe
        )
        rpy2.rinterface_lib.callbacks.consolewrite_print = consolewrite_print_backup
    combat_df = pd.DataFrame(full_combat['data_combat'])

    out = {"Output":buf} # read output

    return combat_df.to_json(date_format='iso', orient='split'), json.dumps(out)

@callback(
    Output('raw-scatter-long','figure'),
    Output('raw-box-long','figure'),
    Output('raw-pca-long','figure'),
    Output('raw-clustergram-long','figure'),
    Output('raw-distplot-norm-long','figure'),
    Output('raw-distplot-kde-long','figure'),
    Input('dropdown-long-batch','value'),
    [Input('dropdown-long-voi','value')],
    Input('dropdown-long-idvar','value'),
    Input('dropdown-long-timevar','value'),
    Input('stored-data-long','data'),
)
def gen_raw_plots(selected_batch,selected_voi,selected_idvar,selected_timevar,stored_data):
    if stored_data == None or selected_batch == None or selected_voi == None \
        or selected_idvar == None or selected_timevar == None:
        raise PreventUpdate
    
    # prep
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    columns = [c for c in selected_voi]
    #columns.append(selected_batch)
    filtered_df = df[columns].apply(pd.to_numeric)
    filtered_df[selected_batch] = df[selected_batch]
    filtered_df[selected_idvar] = df[selected_idvar]
    filtered_df[selected_timevar] = df[selected_timevar]

    # check for missing
    missing_data = False
    for col in filtered_df.columns:
        if any(pd.isnull(filtered_df[col])) and not all(pd.isnull(filtered_df[col])):
            missing_data = True

    # scatter plot
    scatter_plot = px.scatter_matrix(
        filtered_df,
        dimensions=selected_voi,
        color=selected_batch
    )
    scatter_plot.update_traces(diagonal_visible=False)
    scatter_plot.update_layout(title_text='Matrix Scatter Plot')

    # box plot
    box_plot = px.box(filtered_df,color=selected_batch)
    box_plot.update_layout(title_text='Box Plots')

    if missing_data:
        imputed_df = impute_data(filtered_df)

    # PCA
    pca = PCA()
    if missing_data:
        components = pca.fit_transform(imputed_df)
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        pca_plot = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(len(selected_voi)),
            color=imputed_df[selected_batch]
        )
        pca_plot.update_traces(diagonal_visible=False)
        pca_plot.update_layout(title_text='PCA (IMPUTED)')
    else:
        components = pca.fit_transform(filtered_df)
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        pca_plot = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(len(selected_voi)),
            color=filtered_df[selected_batch]
        )
        pca_plot.update_traces(diagonal_visible=False)
        pca_plot.update_layout(title_text='PCA')

    # clustergram
    if missing_data:
        cluster_plot = dash_bio.Clustergram(
            data=imputed_df,
            column_labels=list(imputed_df.columns),
            row_labels=list(imputed_df[selected_batch].values),
            height=800,
            width=700,
            line_width=0,
            hidden_labels="row"
        )
        cluster_plot.update_layout(title_text='Clustergram (IMPUTED)')
    else:
        cluster_plot = dash_bio.Clustergram(
            data=filtered_df,
            column_labels=list(filtered_df.columns),
            row_labels=list(filtered_df[selected_batch].values),
            height=800,
            width=700,
            line_width=0,
            hidden_labels="row"
        )
        cluster_plot.update_layout(title_text='Clustergram')

    # distplots
    if missing_data:
        norm_plot = ff.create_distplot([imputed_df[c] for c in columns],columns, curve_type='normal')
        norm_plot.update_layout(title_text='Normal Distribution Plot (IMPUTED)')

        kde_plot = ff.create_distplot([imputed_df[c] for c in columns],columns)
        kde_plot.update_layout(title_text='KDE Distribution Plot (IMPUTED)')
    else:
        norm_plot = ff.create_distplot([filtered_df[c] for c in columns],columns, curve_type='normal')
        norm_plot.update_layout(title_text='Normal Distribution Plot')

        kde_plot = ff.create_distplot([filtered_df[c] for c in columns],columns)
        kde_plot.update_layout(title_text='KDE Distribution Plot')

    #qqplot_data = filtered_df[selected_voi].to_numpy()
    #qqplot = sm.qqplot(qqplot_data, line='45')
    #buf = io.BytesIO()
    #qqplot.savefig(buf,format='png')
    #qqplot_img_data = base64.b64encode(buf.getbuffer()).decode('ascii')
    #qqplot_img = f'data:image/png;base64,{qqplot_img_data}'
    return scatter_plot, box_plot, pca_plot, cluster_plot, norm_plot, kde_plot, #qqplot_img

@callback(
    Output('combat-scatter-long','figure'),
    Output('combat-box-long','figure'),
    Output('combat-pca-long','figure'),
    Output('combat-clustergram-long','figure'),
    Output('combat-distplot-norm-long','figure'),
    Output('combat-distplot-kde-long','figure'),
    Input('dropdown-long-batch','value'),
    [Input('dropdown-long-voi','value')],
    Input('dropdown-long-idvar','value'),
    Input('dropdown-long-timevar','value'),
    Input('stored-combat-long','data'),
    prevent_initial_call=True 
)
def gen_combat_plots(selected_batch,selected_voi,selected_idvar,selected_timevar,combat_data):
    if combat_data is None or selected_batch is None or selected_voi is None \
        or selected_idvar is None or selected_timevar is None:
        raise PreventUpdate
    
    # prep
    full_df = pd.read_json(io.StringIO(combat_data), orient='split')
    columns = [c for c in selected_voi]
    columns.append(selected_batch)
    full_df.columns = columns
    
    # scatter plot
    scatter_plot = px.scatter_matrix(
        full_df,
        dimensions=selected_voi,
        color=selected_batch
    )
    scatter_plot.update_traces(diagonal_visible=False)
    scatter_plot.update_layout(title_text='Combat Adjusted Scatter Plot')

    # box plot
    box_plot = px.box(full_df,color=selected_batch)
    box_plot.update_layout(title_text='Combat Adjusted Box Plots')

    # PCA
    pca = PCA()
    components = pca.fit_transform(full_df[selected_voi])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    pca_plot = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(len(selected_voi)),
        color=full_df[selected_batch]
    )
    pca_plot.update_traces(diagonal_visible=False)
    pca_plot.update_layout(title_text='Combat Adjusted PCA')

    # clustergram
    cluster_plot = dash_bio.Clustergram(
        data=full_df,
        column_labels=list(full_df.columns),
        row_labels=list(full_df[selected_batch].values),
        height=800,
        width=700,
        line_width=0,
        hidden_labels="row"
    )
    cluster_plot.update_layout(title_text='Combat Adjusted Clustergram')

    # distplots
    norm_plot = ff.create_distplot([full_df[c] for c in selected_voi],selected_voi, curve_type='normal')
    norm_plot.update_layout(title_text='Combat Adjusted Normal Distribution Plot')

    kde_plot = ff.create_distplot([full_df[c] for c in selected_voi],selected_voi)
    kde_plot.update_layout(title_text='Combat Adjusted KDE Distribution Plot')

    #qqplot_data = filtered_df[selected_voi].to_numpy()
    #qqplot = sm.qqplot(qqplot_data, line='45')
    #buf = io.BytesIO()
    #qqplot.savefig(buf,format='png')
    #qqplot_img_data = base64.b64encode(buf.getbuffer()).decode('ascii')
    #qqplot_img = f'data:image/png;base64,{qqplot_img_data}'


    return scatter_plot, box_plot, pca_plot, cluster_plot, norm_plot, kde_plot, #qqplot_img

@callback(
    Output('combat-prior-dist','figure'),
    Input('stored-combat-model','data'),
    prevent_initial_call=True
)
def update_combat_priors(combat_model):
    pass



layout = html.Div([
    html.P(),
    dbc.Tabs([
        dbc.Tab(
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Upload Data", className="card-title"),
                        html.P(
                            "Choose some text data to upload and analyze."
                            " Preferrably CSV data.",
                            className="card-text"
                        ),
                        dcc.Upload(dbc.Button('Upload File',color="primary",class_name="me-1"),
                        id='upload-long-data'),
                        html.Hr(),
                        html.Div(dash_table.DataTable(id='long-data-table'),id="long-data-container")
                    ])
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Define your variables for visualization and analysis.",
                        className="card-title"),
                        html.Hr(),
                        html.P(
                            "Select the unique ID variable."
                        ),
                        dcc.Dropdown(id="dropdown-long-idvar",placeholder="ID variable"),
                        html.Hr(),
                        html.P(
                            "Select the time variable."
                        ),
                        dcc.Dropdown(id='dropdown-long-timevar',placeholder="Time variable"),
                        html.Hr(),
                        html.P(
                            "Select your 'batch' or 'site' grouping variable."
                        ),
                        dcc.Dropdown(id='dropdown-long-batch',placeholder="Batch variable"),
                        html.Hr(),
                        html.P(
                            "Next, select your 'variables of interest.' "
                            "These are the candidate variables for adjustment in Combat."
                        ),
                        dcc.Dropdown(id='dropdown-long-voi',multi=True,
                            placeholder="Variables of Interest"),
                    ])
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.P(
                            "Define the random effects"
                        ),
                        dbc.Input(id="long-combat-ranef",placeholder="(1|SubID)",type="text"),
                    ])
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.P(
                            "Define the statistical model you'd like to use:"
                        ),
                        dbc.Input(id="long-combat-model", placeholder="x+y", type="text"),
                        html.P(),
                        html.Div([
                            dbc.Button("Get Model Matrix",id="long-combat-model-submit",n_clicks=0,className="me-md-2",),
                        ],className="d-grid gap-2 col-6 mx-auto"),
                        html.Div(id='long-combat-model-output')
                    ])
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.P(
                            "Click the 'submit' button to run Combat on your data. "
                            "Upon completion, you can return to the 'Plots' tab to "
                            "visualize your results in comparison to the raw data."
                        ),
                        html.Div([
                            dbc.Button("Submit",id="long-combat-run-submit",n_clicks=0,className="me-md-2"),
                        ],className="d-grid gap-2 col-6 mx-auto"),
                        html.Hr(),
                        html.Div(id="long-combat-run-output") 
                    ])
                )
            ]), label="Setup & Run Longitudinal Combat"
        ),
        dbc.Tab(
            html.Div([
                html.P(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-box-long")),
                            dbc.Col(dcc.Graph(id="combat-box-long"))
                        ])
                    ])
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-scatter-long")),
                            dbc.Col(dcc.Graph(id="combat-scatter-long"))
                        ])
                    ])
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-pca-long")),
                            dbc.Col(dcc.Graph(id="combat-pca-long"))
                        ])
                    ])
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-clustergram-long")),
                            dbc.Col(dcc.Graph(id="combat-clustergram-long"))
                        ])
                    ])
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-distplot-norm-long")),
                            dbc.Col(dcc.Graph(id="combat-distplot-norm-long"))
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="raw-distplot-kde-long")),
                            dbc.Col(dcc.Graph(id="combat-distplot-kde-long"))
                        ])
                    ])
                )
            ]), label="Plots"
        )
    ]),
    dmc.Text(id="long-txt"),
    dcc.Store(id="stored-long-data",storage_type="session"),
    dcc.Store(id="stored-long-combat",storage_type="memory"),
    dcc.Store(id="stored-long-model",storage_type="memory"),
    dcc.Store(id="long-combat-stdout",storage_type="memory"),
])