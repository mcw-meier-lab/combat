import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_extensions.enrich import (
    callback, Output, Input, State,
    dcc, html, dash_table, DashLogger, ctx, ALL
)
from dash.exceptions import PreventUpdate

import io
import json
import base64
import pandas as pd
#import seaborn as sns
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from formulaic import Formula
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
        dbc.Button("Download Adjusted Data",id="btn-long-download",color="primary",n_clicks=0),
        dcc.Download(id="download-long-combat-csv"),
        html.Hr()
    ],id="long-download-btn-container",className="d-grid gap-2 col-6 mx-auto")
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        page_size=20,
    )], style={'width':'98%'})

    card = dbc.Card(
        dbc.CardBody([button,table])
    )
    
    return card, ""


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
    Output("plots","children"),
    Input("dropdown-long-batch","value"),
    [Input("dropdown-long-voi","value")],
    Input("dropdown-long-idvar","value"),
    Input("dropdown-long-timevar","value"),
    Input("stored-long-data","data"),
    Input("stored-long-combat","data")
)
def gen_plots(selected_batch,selected_voi,selected_idvar,selected_timevar,long_data,combat_data):
    if selected_batch is None or selected_voi is None or selected_idvar is None or selected_timevar is None or long_data is None:
        raise PreventUpdate

    long_df = pd.read_json(io.StringIO(long_data), orient='split')
    filtered_df = long_df[selected_voi].apply(pd.to_numeric)
    filtered_df[selected_batch] = long_df[selected_batch]
    filtered_df[selected_idvar] = long_df[selected_idvar]
    filtered_df[selected_timevar] = long_df[selected_timevar]

    missing_data = False
    for col in filtered_df.columns:
        if any(pd.isnull(filtered_df[col])) and not all(pd.isnull(filtered_df[col])):
            missing_data = True

    avg_df = pd.DataFrame(filtered_df.groupby([selected_idvar,selected_batch,selected_timevar])[selected_voi].mean()).reset_index()
    groups = list(avg_df[selected_batch].unique())

    box_plot = px.box(avg_df[selected_voi + [selected_batch]],color=selected_batch,points="all")
    box_plot.update_layout(title_text="Box Plots")

    scatter_plot = px.scatter_matrix(
        avg_df,
        dimensions=selected_voi,
        color=selected_batch
    )
    scatter_plot.update_traces(diagonal_visible=False)
    scatter_plot.update_layout(title_text='Matrix Scatter Plot')

    if missing_data:
        imputed_df = impute_data(filtered_df) 
        avg_df = pd.DataFrame(imputed_df.groupby([selected_idvar,selected_batch,selected_timevar])[selected_voi].mean()).reset_index()

    pca = PCA()
    components = pca.fit_transform(avg_df[selected_voi])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    pca_plot = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(len(selected_voi)),
        color=avg_df[selected_batch]
    )
    pca_plot.update_traces(diagonal_visible=False)
    pca_plot.update_layout(title_text='PCA')

    # distplots
    norm_plots = []
    kde_plots = []
    for voi in range(len(selected_voi)):
        norm = ff.create_distplot(
            [avg_df[avg_df[selected_batch] == g][selected_voi[voi]] for g in groups],
            groups,curve_type='normal')
        norm.update_layout(title_text=f'{selected_voi[voi]} Normal Distribution Plot')
        norm_plots.append(norm)

        kde = ff.create_distplot(
            [avg_df[avg_df[selected_batch] == g][selected_voi[voi]] for g in groups],groups)
        kde.update_layout(title_text=f'{selected_voi[voi]} KDE Distribution Plot')
        kde_plots.append(kde)


    if combat_data:
        combat_df = pd.read_json(io.StringIO(combat_data),orient='split')
        combat_voi = [f"{c}.combat" for c in selected_voi]
        combat_avg = pd.DataFrame(combat_df.groupby([selected_idvar,selected_batch,selected_timevar])[combat_voi].mean()).reset_index()

    if not combat_data:
        combat_box = go.Figure()
        combat_scatter = go.Figure()
        combat_pca = go.Figure()
        combat_norms = [go.Figure() for voi in range(len(selected_voi))]
        combat_kdes = [go.Figure() for voi in range(len(selected_voi))]
    else:
        combat_box = px.box(combat_avg[combat_voi + [selected_batch]],color=selected_batch,points="all")
        combat_box.update_layout(title_text="Combat Adjusted Box Plot")

        combat_scatter = px.scatter_matrix(
            combat_avg,dimensions=combat_voi,color=selected_batch
        )
        combat_scatter.update_traces(diagonal_visible=False)
        combat_scatter.update_layout(title_text="Combat Adjusted Matrix Scatter Plot")

        components = pca.fit_transform(combat_avg[combat_voi])
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        combat_pca = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(len(combat_voi)),
            color=combat_avg[selected_batch]
        )
        combat_pca.update_traces(diagonal_visible=False)
        combat_pca.update_layout(title_text='Combat Adjusted PCA')

        combat_norms = []
        combat_kdes = []
        for voi in range(len(combat_voi)):
            c_norm = ff.create_distplot(
                [combat_avg[combat_avg[selected_batch] == g][combat_voi[voi]] for g in groups],groups, curve_type='normal')
            c_norm.update_layout(title_text=f'{combat_voi[voi]} Combat Adjusted Normal Distribution Plot')
            combat_norms.append(c_norm)

            c_kde = ff.create_distplot([combat_avg[combat_avg[selected_batch] == g][combat_voi[voi]] for g in groups],groups)
            c_kde.update_layout(title_text=f'{combat_voi[voi]} Combat Adjusted KDE Distribution Plot')
            combat_kdes.append(c_kde)


    plots = html.Div([
        html.P(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=box_plot)),
                    dbc.Col(dcc.Graph(figure=combat_box))
                ])
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=scatter_plot)),
                    dbc.Col(dcc.Graph(figure=combat_scatter))
                ])
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=pca_plot)),
                    dbc.Col(dcc.Graph(figure=combat_pca))
                ])
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=norm_plots[ii])),
                    dbc.Col(dcc.Graph(figure=combat_norms[ii]))
                ]) for ii in range(len(selected_voi))
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=kde_plots[ii])),
                    dbc.Col(dcc.Graph(figure=combat_kdes[ii]))
                ]) for ii in range(len(selected_voi))
            ])
        )

    ],id="plots")

    return plots

#@callback(
#    Output("traj","children"),
#    Input("dropdown-long-batch","value"),
#    [Input("dropdown-long-voi","value")],
#    Input("dropdown-long-idvar","value"),
#    Input("dropdown-long-timevar","value"),
#    Input("stored-long-data","data"),
#    Input("stored-long-combat","data")
#)
#def gen_trajectories(selected_batch,selected_voi,selected_idvar,selected_timevar,long_data,combat_data):
#    if selected_batch is None or selected_voi is None or selected_idvar is None or selected_timevar is None or long_data is None:
#        raise PreventUpdate
#
#    long_df = pd.read_json(io.StringIO(long_data), orient='split')
#    filtered_df = long_df[selected_voi].apply(pd.to_numeric)
#    filtered_df[selected_batch] = long_df[selected_batch]
#    filtered_df[selected_idvar] = long_df[selected_idvar]
#    filtered_df[selected_timevar] = long_df[selected_timevar]
#
#    missing_data = False
#    for col in filtered_df.columns:
#        if any(pd.isnull(filtered_df[col])) and not all(pd.isnull(filtered_df[col])):
#            missing_data = True
#
#    avg_df = pd.DataFrame(filtered_df.groupby([selected_idvar,selected_batch,selected_timevar])[selected_voi].mean()).reset_index()
#
#    traj_plots = []
#    for voi in range(len(selected_voi)):
#        traj = sns.catplot(kind='point',
#           data=avg_df,
#           x=selected_timevar,
#           y=selected_voi[voi],
#           hue=selected_batch
#        )
        
    
#    return

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
                        html.Div(id="btn-long-download-container"),
                        html.Hr(),
                        html.Div(id="long-combat-run-output") 
                    ])
                )
            ]),
            label="Setup & Run Longitudinal Combat",
            tab_id="setup_tab"
        ),
        dbc.Tab(html.Div(id="plots",children=[]),
            label="Plots",
            tab_id="plots_tab"
        ),
#        dbc.Tab(html.Div(id="traj",children=[]),
#            label="Trajectories",
#            tab_id="traj_tab"
#        )
    ],
    active_tab="setup_tab"),
    dmc.Text(id="long-txt"),
    dcc.Store(id="stored-long-data",storage_type="session"),
    dcc.Store(id="stored-long-combat",storage_type="memory"),
    dcc.Store(id="stored-long-model",storage_type="memory"),
    dcc.Store(id="long-combat-stdout",storage_type="memory"),
])