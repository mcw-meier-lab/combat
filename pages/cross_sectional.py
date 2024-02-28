import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_extensions.enrich import (
    html, Output, Input, 
    State, callback, dash_table,
    dcc, DashLogger, ctx
)
from dash.exceptions import PreventUpdate

import base64
import io
import json
import pandas as pd
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from formulaic import Formula
from neuroHarmonize import harmonizationLearn

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import dash_bio
from sklearn.decomposition import PCA

from config import impute_data

dash.register_page(
    __name__,
    path='/cross_sectional',
    title='Cross-Sectional Combat',
    name='Cross-Sectional Combat'
)

@callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
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
    Output('data-container', 'children'),
    Input('stored-data', 'data'),
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
        id="data-table"
    )], style={'width':'98%'},id="data-container")
    
    return table


@callback(
    Output('dropdown-voi','options'),
    Output('dropdown-voi','value'),
    Input('dropdown-batch','value'),
    Input('data-table','selected_columns'),
    Input('stored-data','data'),
    prevent_initial_call=True
)
def set_setup_voi(selected_batch,selected_cols,stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    voi_options = [c for c in df.columns if c != selected_batch]
    voi_value = []
    if selected_cols:
        voi_value = [c for c in selected_cols if c != selected_batch]
        voi_options = [c for c in selected_cols if c != selected_batch]

    return voi_options, voi_value

@callback(
    Output('dropdown-batch','options'),
    Input('stored-data','data'),
)
def set_setup_batch(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    batch_options = [c for c in df.columns]

    return batch_options

@callback(
    Output('combat-model-output','children'),
    Input('stored-data','data'),
    State('combat-model','value'),
    Input('combat-model-submit','n_clicks'),
    prevent_initial_call=True
)
def get_model_matrix(stored_data,combat_model,n_clicks):
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

## Run Cross-sectional Combat ##
@callback(
    Output("combat-version-options","children"),
    Input("combat-run-version","value"),
    Input('dropdown-batch','value'),
    [Input('dropdown-voi','value')],
    Input('stored-data','data'),
    prevent_initial_call=True
)
def update_combat_options(combat_version,selected_batch,selected_voi,stored_data):
    if combat_version is None:
        raise PreventUpdate
    if combat_version == 1: #Fortin
        alert = html.Div([
            dbc.Alert(
                [
                    "This version of Combat does not handle missing data. You can still run it, but data will be imputed via scikit-learn's Iterative Imputer (MICE algorithm): ",
                    html.A("See documentation",href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html")
                ],
                color="warning")
        ])
        extra_options = html.Div([
            # run without empirical bayes
            dbc.Label("This version of Combat runs with Empirical Bayes but you can disable this option below: ",html_for="switch-eb"),
            dmc.Switch(checked=False,id="switch-eb"),
            dbc.Label("Additionally, use non-parametric adjustments (default is parametric): ",html_for="switch-parametric"),
            dmc.Switch(checked=False,id="switch-parametric"),
            html.Div(dcc.Dropdown(id="dropdown-nonlinear"),style={"display":"none"})
        ])
        options = [alert, extra_options]
    elif combat_version == 2: #Pomponio
        df = pd.read_json(io.StringIO(stored_data), orient='split')
        covars = [c for c in df.columns \
                  if c != selected_batch or c not in selected_voi]
        alert = html.Div([
            dbc.Alert(
                [
                    "This version of Combat does not handle missing data. You can still run it, but data will be imputed via scikit-learn's Iterative Imputer (MICE algorithm): ",
                    html.A("See documentation",href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html")
                ],
                color="warning")
        ])
        extra_options = html.Div([
            # run without empirical bayes
            dbc.Label("This version of Combat runs with Empirical Bayes but you can disable this option below: ",html_for="switch-eb"),
            dmc.Switch(checked=False,id="switch-eb"),
            dbc.Label("You can also define non-linear covariates below:",
                      html_for="dropdown-nonlinear"),
            dcc.Dropdown(
                options=covars,
                id='dropdown-nonlinear',
                placeholder="Non-linear variable(s)"
            ),
            html.Div(dmc.Switch(id="switch-parametric"),style={"display":"none"})
        ])
        options = [alert,extra_options]
    elif combat_version == 3: #ENIGMA
        extra_options = html.Div([
            # run without empirical bayes
            dbc.Label("This version of Combat runs with Empirical Bayes but you can disable this option below: ",html_for="switch-eb"),
            dmc.Switch(checked=False,id="switch-eb"),
            html.Div([dcc.Dropdown(id="dropdown-nonlinear"),dmc.Switch(id="switch-parametric")],style={"display":"none"})
        ])
        options = [extra_options]
    return dbc.Form(options,id="combat-version-options")

@callback(
        Output("download-combat-csv","data"),
        Input("btn-download","n_clicks"),
        Input("stored-combat","data"),
        prevent_initial_call=True
)
def download_combat_table(n_clicks,stored_combat):
    if stored_combat is None or n_clicks == 0:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_combat), orient='split') 
    return dcc.send_data_frame(df.to_csv,"combat_output.csv")

@callback(
    Output("combat-run-output","children"),
    Output("txt","children"),
    Input("stored-combat","data"),
    Input("combat-run-version","value"),
    Input('combat-run-submit','n_clicks'),
    Input("combat-stdout","data"),
    prevent_initial_call=True,
    log=True
)
def update_combat_table(stored_combat,combat_version,n_clicks,combat_stdout,dash_logger: DashLogger):
    if stored_combat is None:
        raise PreventUpdate
    if ctx.triggered_id != "combat-run-submit":
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_combat), orient='split')
    if combat_version == 1:
        ver = "Fortin's Combat"
    elif combat_version == 2:
        ver = "Pomponio's Combat"
    elif combat_version == 3:
        ver = "ENIGMA Combat"
    output = "".join(json.loads(combat_stdout)["Output"])
    dash_logger.info(message=output,title=f"{ver} finished!")

    button = html.Div([
        dbc.Button("Download Adjusted Data",id="btn-download",color="primary",n_clicks=0,className="me-md-2"),
        dcc.Download(id="download-combat-csv"),
        html.Hr()
    ], id="btn-container",className="d-grid gap-2 col-6 mx-auto")
    table = html.Div([
                dash_table.DataTable(
                    df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'scroll'},
                    page_size=20,
                    id="output-table"
                )], style={'width':'98%'},id="combat_run_output")
    card = dbc.Card(
        dbc.CardBody([button,table],className="mt-3")
    )
    
    return card, ""


@callback(
    Output("stored-combat","data"),
    Output("stored-combat-model","data"),
    Output("combat-stdout","data"),
    Input("dropdown-batch","value"),
    [Input("dropdown-voi","value")],
    State('combat-model','value'),
    Input("combat-run-version","value"),
    [
        Input("switch-eb","on"),
        [Input("dropdown-nonlinear","value")],
        Input("switch-parametric","on"),
    ],
    Input('combat-run-submit','n_clicks'),
    Input("stored-data","data"),
    prevent_initial_call=True,
    log=True
)
def run_combat(selected_batch,selected_voi,combat_model,combat_version,switch_eb,selected_nonlinear,switch_nonparametric,n_clicks,stored_data,dash_logger=DashLogger):
    if stored_data is None or combat_model is None or selected_batch is None \
        or selected_voi is None or combat_version is None:
        raise PreventUpdate
    if ctx.triggered_id != "combat-run-submit":
        raise PreventUpdate
    
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    model_matrix = Formula(combat_model).get_model_matrix(df)
    columns = [c for c in selected_voi]

    voi_data = df[columns].apply(pd.to_numeric)
    voi_data[selected_batch] = df[selected_batch]

    # check for missing
    missing_data = False
    for col in voi_data.columns:
        if any(pd.isnull(voi_data[col])) and not all(pd.isnull(voi_data[col])):
            missing_data = True

    # check for factored vars
    use_factor = False
    factor_vars = []
    for c in selected_voi:
        if c.lower() == "sex" or "sex" in c.lower():
           factor_var = pd.get_dummies(df[c],dtype=float)
           df[f"{c}_factor"] = factor_var
           use_factor = True
           factor_vars.append(c)


    if combat_version == 1: #Fortin neuroCombat
        # run R version
        # neuroCombat(dat=t(ln_data),batch=batch)
        #with conversion.localconverter(default_converter):
        buf = []

        if missing_data:
            voi_data = impute_data(voi_data)
        with (ro.default_converter + pandas2ri.converter).context():
            r_dataframe = ro.conversion.get_conversion().py2rpy(voi_data[selected_voi].T)
            r_eb = ro.BoolVector([not switch_eb])
            r_parametric = ro.BoolVector([not switch_nonparametric])
            neuroCombat = importr('neuroCombat')
            consolewrite_print_backup = rpy2.rinterface_lib.callbacks.consolewrite_print
            rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: buf.append(x)
            full_combat = neuroCombat.neuroCombat(
                r_dataframe,voi_data[selected_batch],model_matrix,eb=r_eb,parametric=r_parametric
            )
            rpy2.rinterface_lib.callbacks.consolewrite_print = consolewrite_print_backup
        combat_df = pd.DataFrame(full_combat['dat.combat']).T
        combat_df.columns = columns
        combat_df[selected_batch] = df[selected_batch]
        model = pd.DataFrame(full_combat['estimates']['gamma.hat'])
        model['gamma.bar'] = full_combat['estimates']['gamma.bar']
        model['t2'] = full_combat['estimates']['t2']
            

    elif combat_version == 2: #Pomponio's neuroHarmonize
        # run python version
        # ncovar -> SITE plus whatever covariates
        # smooth_terms -> NONLINEAR covariates
        # apply model???
        if missing_data:
            voi_data = impute_data(voi_data)
        if use_factor:
            for var in factor_vars:
                voi_data[var.replace("_","")] = df[var]
            
        voi_data = voi_data.rename(columns={selected_batch:"SITE"})
        ncovar = pd.DataFrame(voi_data['SITE']).apply(lambda x: x.str.replace(" ","_") if type(x) == str else x)
        model_matrix = pd.DataFrame(model_matrix)
        model_matrix['SITE'] = ncovar
        model_matrix = model_matrix.drop("Intercept",axis=1)
        if not selected_nonlinear is None:
            model_matrix[selected_nonlinear] = df[selected_nonlinear]
            full_model,adjusted = harmonizationLearn(voi_data[columns].to_numpy(),model_matrix,eb=not switch_eb,smooth_terms=selected_nonlinear)
        else:
            full_model,adjusted = harmonizationLearn(voi_data[columns].to_numpy(),model_matrix,eb=not switch_eb)
        combat_df = pd.DataFrame(adjusted)
        combat_df.columns = columns
        combat_df[selected_batch] = df[selected_batch]
        model = pd.DataFrame(full_model['gamma_hat'])
        model['gamma_bar'] = full_model['gamma_bar']
        model['t2'] = full_model['t2']
        buf = f"[neuroHarmonize] Ran with Empirical Bayes: {not switch_eb}\n" \
            f"[neuroHarmonize] smooth_terms: {selected_nonlinear}"


    elif combat_version == 3: #ENIGMA Combat
        buf = []
        # run R version
        with (ro.default_converter + pandas2ri.converter).context():
            consolewrite_print_backup = rpy2.rinterface_lib.callbacks.consolewrite_print
            rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: buf.append(x)
            dat_df = ro.conversion.get_conversion().py2rpy(voi_data[selected_voi])
            batch_vec = ro.FactorVector(ro.conversion.get_conversion().py2rpy(voi_data[selected_batch]))
            r_eb = ro.BoolVector([not switch_eb])
            r = ro.r
            r['source']("enigma.R")
            enigma_func = ro.globalenv['run_enigma_combat']
            enigma_res = enigma_func(
                dat_df,batch_vec,model_matrix,r_eb)
            rpy2.rinterface_lib.callbacks.consolewrite_print = consolewrite_print_backup
        combat_df = pd.DataFrame(enigma_res['dat.combat'])
        combat_df.columns = columns
        combat_df[selected_batch] = df[selected_batch]
        model = pd.DataFrame(enigma_res['gamma.hat'])
        model['gamma.bar'] = enigma_res['gamma.bar']
        model['t2'] = enigma_res['t2']


    out = {"Output":buf} # read output
    return combat_df.to_json(date_format='iso', orient='split'),model.to_json(date_format="iso",orient="split",default_handler=str), json.dumps(out)

@callback(
    Output("plots","children"),
    Input("dropdown-batch","value"),
    [Input("dropdown-voi","value")],
    Input("stored-data","data"),
    Input("combat-run-version","value"),
    Input("stored-combat","data")
)
def gen_plots(selected_batch,selected_voi,stored_data,combat_version,combat_data):
    if selected_batch is None or selected_voi is None or stored_data is None:
        raise PreventUpdate
    
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    filtered_df = df[selected_voi].apply(pd.to_numeric)
    filtered_df[selected_batch] = df[selected_batch]

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
        norm_plot = ff.create_distplot([imputed_df[c] for c in selected_voi],selected_voi, curve_type='normal')
        norm_plot.update_layout(title_text='Normal Distribution Plot (IMPUTED)')

        kde_plot = ff.create_distplot([imputed_df[c] for c in selected_voi],selected_voi)
        kde_plot.update_layout(title_text='KDE Distribution Plot (IMPUTED)')
    else:
        norm_plot = ff.create_distplot([filtered_df[c] for c in selected_voi],selected_voi, curve_type='normal')
        norm_plot.update_layout(title_text='Normal Distribution Plot')

        kde_plot = ff.create_distplot([filtered_df[c] for c in selected_voi],selected_voi)
        kde_plot.update_layout(title_text='KDE Distribution Plot')

    # Combat plots
    if not combat_data:
        ver = ""
        combat_box = go.Figure()
        combat_scatter = go.Figure()
        combat_pca = go.Figure()
        combat_cluster = go.Figure()
        combat_norm = go.Figure()
        combat_kde = go.Figure()
    else:
        combat_df = pd.read_json(io.StringIO(combat_data), orient='split')
        combat_df.columns = selected_voi + [selected_batch]

        if combat_version == 1:
            ver="Fortin"
        elif combat_version == 2:
            ver="Pomponio"
        elif combat_version == 3:
            ver="ENIGMA"
    
        combat_box = px.box(combat_df,color=selected_batch)
        combat_box.update_layout(title_text=f'{ver} Combat Adjusted Box Plots')

        combat_scatter = px.scatter_matrix(
            combat_df,
            dimensions=selected_voi,
            color=selected_batch
        )
        combat_scatter.update_traces(diagonal_visible=False)
        combat_scatter.update_layout(title_text=f'{ver} Combat Adjusted Scatter Plot')

        components = pca.fit_transform(combat_df[selected_voi])
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        combat_pca = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(len(selected_voi)),
            color=combat_df[selected_batch]
        )
        combat_pca.update_traces(diagonal_visible=False)
        combat_pca.update_layout(title_text=f'{ver} Combat Adjusted PCA')

        combat_cluster = dash_bio.Clustergram(
            data=combat_df,
            column_labels=list(combat_df.columns),
            row_labels=list(combat_df[selected_batch].values),
            height=800,
            width=700,
            line_width=0,
            hidden_labels="row"
        )
        combat_cluster.update_layout(title_text=f'{ver} Combat Adjusted Clustergram')

        combat_norm = ff.create_distplot([combat_df[c] for c in selected_voi],selected_voi, curve_type='normal')
        combat_norm.update_layout(title_text=f'{ver} Combat Adjusted Normal Distribution Plot')

        combat_kde = ff.create_distplot([combat_df[c] for c in selected_voi],selected_voi)
        combat_kde.update_layout(title_text=f'{ver} Combat Adjusted KDE Distribution Plot')

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
                    dbc.Col(dcc.Graph(figure=cluster_plot)),
                    dbc.Col(dcc.Graph(figure=combat_cluster))
                ])
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=norm_plot)),
                    dbc.Col(dcc.Graph(figure=combat_norm))
                ])
            ])
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=kde_plot)),
                    dbc.Col(dcc.Graph(figure=combat_kde))
                ])
            ])
        )
    ])

    return plots

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
                        dcc.Upload(dbc.Button('Upload File',color="primary",class_name="me-1"),id='upload-data'
                        ),
                        html.Hr(),
                        html.Div(dash_table.DataTable(id='data-table'),id="data-container"),
                    ])
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Define your variables for visualization and analysis.",className="card-title"
                        ),
                        html.Hr(),
                        html.P(
                            "First, select your 'batch' or 'site' grouping variable."
                        ),
                        dcc.Dropdown(id='dropdown-batch',placeholder="Batch variable"),
                        html.Hr(),
                        html.P(
                            "Next, select your 'variables of interest.' "
                            "These are the candidate variables for adjustment in Combat."
                        ),
                        dcc.Dropdown(id='dropdown-voi',multi=True,
                            placeholder="Variables of Interest"),
                    ]),
                    className="mt-3"
                ),
                dbc.Card(
                    dbc.CardBody([
                        html.H4(
                            "Choose your Combat version:",className="card-title"
                        ),
                        dbc.RadioItems(options=[
                            {"label":"Fortin's neuroCombat","value":1},
                            {"label":"Pomponio's neuroHarmonize","value":2},
                            {"label":"ENIGMA Combat","value":3}
                            ],
                            value=1,
                            id="combat-run-version"
                        ),
                        html.Div([html.P(id="combat-version-options"),html.Div([dmc.Switch(id="switch-eb"),dmc.Switch(id="switch-parametric"),dcc.Dropdown(id="dropdown-nonlinear")],style={'display':'none'})])
                    ]),
                    className="mt-3"
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        html.H4(
                            "Define the statistical model you'd like to use:",
                            className="card-title"
                        ),
                        dbc.Input(id="combat-model", placeholder="~x+y", type="text"),
                        html.P(),
                        html.Div([
                            dbc.Button("Get Model Matrix",id="combat-model-submit",n_clicks=0,className="me-md-2",),
                        ],className="d-grid gap-2 col-6 mx-auto"),
                        html.Div(id='combat-model-output')
                    ]),
                    className="mt-3"
                ),
                html.Hr(),
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Run Combat!",className="card-title"),
                        html.P(
                            "Click the 'submit' button to run Combat on your data. "
                            "Upon completion, you can return to the 'Plots' tab to "
                            "visualize your results in comparison to the raw data."
                        ),
                        html.Div([
                            dbc.Button("Submit",id="combat-run-submit",n_clicks=0,className="me-md-2"),
                            ],className="d-grid gap-2 col-6 mx-auto"),
                        html.Div(id="btn-container"),
                        html.Hr(),
                        html.Div(id="combat-run-output")
                    ]),
                    className="mt-3"
                )
            ]),
            label="Setup & Run Combat",
            tab_id="setup_tab"
        ),
        dbc.Tab(
            html.Div(id="plots",children=[]),
            label="Plots",
            tab_id="plots_tab"
        )
    ],active_tab="setup_tab"),
    dmc.Text(id="txt"),
    dcc.Store(id="stored-data",storage_type="session"),
    dcc.Store(id="stored-combat",storage_type="memory"),
    dcc.Store(id="stored-combat-model",storage_type="memory"),
    dcc.Store(id="combat-stdout",storage_type="memory"),
])

