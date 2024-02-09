import base64
import io
import json
import dash_bio
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from formulaic import Formula
#from scipy import stats
from neuroHarmonize import harmonizationLearn
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from sklearn.linear_model import LinearRegression
#from statsmodels.graphics import gofplots as gof
#import statsmodels.api as sm
#from dash import (
#    Dash, html, dcc, 
#    dash_table, Input, 
#    Output, State
#)
from dash import dcc, dash_table, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib
matplotlib.use('agg')
import dash_mantine_components as dmc
from dash_extensions.enrich import Output, Input, html, DashProxy, LogTransform, DashLogger, State



#### Helper function for missing data
def impute_data(df):
    is_missing = df[[col for col in df.columns
                     if any(pd.isnull(df[col])) and not all(pd.isnull(df[col]))]]
    imputer = IterativeImputer(missing_values=np.nan)
    imputed = imputer.fit_transform(is_missing)
    imputed_df = pd.DataFrame(imputed,columns=is_missing.columns)
    merged_df = df.combine_first(imputed_df)

    return merged_df

app = DashProxy(__name__,external_stylesheets=[dbc.themes.FLATLY,dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True,transforms=[LogTransform()], prevent_initial_callbacks=True)
#app = Dash(__name__,external_stylesheets=[dbc.themes.FLATLY,dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)



@app.callback(
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
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
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

    #return html.Div([dcc.Store(id='stored-data', data=df.to_dict('records'))])
    return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('output-data-upload', 'children'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def output_from_store(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    #table = html.Div([dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, responsive='sm',style_table={'overflowX': 'scroll'})],style={'width':1000})
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        page_size=20,
    )], style={'width':'98%'})
    
    return table


@app.callback(
    Output('dropdown-voi','options'),
    Input('dropdown-batch','value'),
    Input('stored-data','data'),
)
def set_setup_voi(selected_batch,stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    voi_options = [c for c in df.columns if c != selected_batch]

    return voi_options

@app.callback(
    Output('dropdown-batch','options'),
    Input('stored-data','data'),
)
def set_setup_batch(stored_data):
    if stored_data is None:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    batch_options = [c for c in df.columns]

    return batch_options

@app.callback(
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

@app.callback(
    Output('raw-scatter','figure'),
    Output('raw-box','figure'),
    Output('raw-pca','figure'),
    Output('raw-clustergram','figure'),
    Output('raw-distplot-norm','figure'),
    Output('raw-distplot-kde','figure'),
    #Output('raw-qqplot','src'),
    Input('dropdown-batch','value'),
    [Input('dropdown-voi','value')],
    Input('stored-data','data'),
    prevent_initial_call=True,
)
def gen_raw_plots(selected_batch,selected_voi,stored_data):
    if stored_data == None or selected_batch == None or selected_voi == None:
        raise PreventUpdate
    
    # prep
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    columns = [c for c in selected_voi]
    #columns.append(selected_batch)
    filtered_df = df[columns].apply(pd.to_numeric)
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

## Run Combat ##
@app.callback(
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
            daq.BooleanSwitch(on=False,id="switch-eb"),
            dbc.Label("Additionally, use non-parametric adjustments (default is parametric): ",html_for="switch-parametric"),
            daq.BooleanSwitch(on=False,id="switch-parametric"),
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
            daq.BooleanSwitch(on=False,id="switch-eb"),
            dbc.Label("You can also define non-linear covariates below:",
                      html_for="dropdown-nonlinear"),
            dcc.Dropdown(
                options=covars,
                id='dropdown-nonlinear',
                placeholder="Non-linear variable(s)"
            ),
            html.Div(daq.BooleanSwitch(id="switch-parametric"),style={"display":"none"})
        ])
        options = [alert,extra_options]
    elif combat_version == 3: #ENIGMA
        extra_options = html.Div([
            # run without empirical bayes
            dbc.Label("This version of Combat runs with Empirical Bayes but you can disable this option below: ",html_for="switch-eb"),
            daq.BooleanSwitch(on=False,id="switch-eb"),
            html.Div([dcc.Dropdown(id="dropdown-nonlinear"),daq.BooleanSwitch(id="switch-parametric")],style={"display":"none"})
        ])
        options = [extra_options]
    return dbc.Form(options,id="combat-version-options")

@app.callback(
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

@app.callback(
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
        dbc.Button("Download",id="btn-download",color="primary",n_clicks=0),
        dcc.Download(id="download-combat-csv")
    ])
    table = html.Div([
    dash_table.DataTable(
        df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        page_size=20,
    )], style={'width':'98%'})
    
    return [table,button]


@app.callback(
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
            r =  ro.r
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

@app.callback(
    Output('combat-prior-dist','figure'),
    Input('stored-combat-model','data'),
    prevent_initial_call=True
)
def update_combat_priors(combat_model):
    pass


@app.callback(
    Output('combat-scatter','figure'),
    Output('combat-box','figure'),
    Output('combat-pca','figure'),
    Output('combat-clustergram','figure'),
    Output('combat-distplot-norm','figure'),
    Output('combat-distplot-kde','figure'),
    #Output('combat-qqplot','src'),
    Input('dropdown-batch','value'),
    [Input('dropdown-voi','value')],
    Input('stored-combat','data'),
    Input("combat-run-version","value"),
    prevent_initial_call=True 
)
def gen_combat_plots(selected_batch,selected_voi,combat_data,combat_version):
    if combat_data is None or selected_batch is None or selected_voi is None:
        raise PreventUpdate
    
    # prep
    full_df = pd.read_json(io.StringIO(combat_data), orient='split')
    columns = [c for c in selected_voi]
    columns.append(selected_batch)
    full_df.columns = columns

    if combat_version == 1:
        ver="Fortin"
    elif combat_version == 2:
        ver="Pomponio"
    elif combat_version == 3:
        ver="ENIGMA"
    

    # scatter plot
    scatter_plot = px.scatter_matrix(
        full_df,
        dimensions=selected_voi,
        color=selected_batch
    )
    scatter_plot.update_traces(diagonal_visible=False)
    scatter_plot.update_layout(title_text=f'{ver} Combat Adjusted Scatter Plot')

    # box plot
    box_plot = px.box(full_df,color=selected_batch)
    box_plot.update_layout(title_text=f'{ver} Combat Adjusted Box Plots')

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
    pca_plot.update_layout(title_text=f'{ver} Combat Adjusted PCA')

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
    cluster_plot.update_layout(title_text=f'{ver} Combat Adjusted Clustergram')

    # distplots
    norm_plot = ff.create_distplot([full_df[c] for c in selected_voi],selected_voi, curve_type='normal')
    norm_plot.update_layout(title_text=f'{ver} Combat Adjusted Normal Distribution Plot')

    kde_plot = ff.create_distplot([full_df[c] for c in selected_voi],selected_voi)
    kde_plot.update_layout(title_text=f'{ver} Combat Adjusted KDE Distribution Plot')

    #qqplot_data = filtered_df[selected_voi].to_numpy()
    #qqplot = sm.qqplot(qqplot_data, line='45')
    #buf = io.BytesIO()
    #qqplot.savefig(buf,format='png')
    #qqplot_img_data = base64.b64encode(buf.getbuffer()).decode('ascii')
    #qqplot_img = f'data:image/png;base64,{qqplot_img_data}'

    return scatter_plot, box_plot, pca_plot, cluster_plot, norm_plot, kde_plot, #qqplot_img


## Pages/Tab Layouts ##
header = html.Div(
    dbc.Container(
        [
            html.H1("Data Harmonization and Batch Effect Analysis", className="display-3"),
            html.P(
                "This web app is designed to facilitate the "
                "analysis and visualization of data that may "
                "be affected by batch or site differences. "
                "You can upload your dataset, view various plots, "
                "run multiple versions of Combat (depending on your data), "
                "and compare results. See the steps below to get started!",
                className="lead",
            ),
            html.Hr(className="my-2"),
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3",
)

steps = html.Div(
    dbc.Accordion([
        dbc.AccordionItem([
            html.P(
                "Head to the 'Upload' tab to upload and preview your data. "
                "This should be text data, preferrably in CSV format."
            ),
        ], 
        title="Step 1"),
        dbc.AccordionItem([
            html.P(
                "Head to the 'Setup' tab to choose your "
                "'variables of interest' "
                "(those variables you may want to adjust via Combat), "
                "your 'batch' or site variable, and any additional "
                "'condition' variables or covariates you want to account for."
            )
        ],
        title="Step 2"),
        dbc.AccordionItem([
            html.P(
                "Decide which version of Combat to run. "
                "You can use the 'Plots' tab to visualize your "
                "data (once it's been uploaded) and help your "
                "decision. Additionally, the links below can "
                "help you decide which flavor of Combat to use. "
                "When you're ready, head to the correct 'Combat' tab!"
            ),
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "Warning! Not every version of Combat is suited to working "
                "with missing data."],
                color="warning",
                className="d-flex align-items-center"),
            dbc.Row([
                dbc.Col([
                    dcc.Link("Fortin's neuroCombat",href="https://github.com/Jfortin1/neuroCombat_Rpackage/tree/master"),
                ]),
                dbc.Col([
                    dcc.Link("Pomponio's neuroHarmonize",href="https://github.com/rpomponio/neuroHarmonize"),
                ]),
                dbc.Col([
                    dcc.Link("ENIGMA Combat",href="https://cran.r-project.org/web/packages/combat.enigma/index.html"),
                ])
            ]),
        ],
        title="Step 3")
    ], start_collapsed=True)
)
tab1_upload = dbc.Card(
    dbc.CardBody([
        html.H4("Upload Data", className="card-title"),
        html.P(
            "Choose some text data to upload and analyze."
             " Preferrably CSV data.",
             className="card-text"
        ),
        dcc.Upload(dbc.Button('Upload File',color="primary",class_name="me-1"),
               id='upload-data'),
        html.Hr(),
        html.Div(id='output-data-upload'),
    ])
)

tab2_plots = html.Div([
    dbc.Card(
        dbc.CardBody([
            # box plots
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-box')),
                dbc.Col(dcc.Graph(id='combat-box'))
            ])
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            # scatter plots
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-scatter')),
                dbc.Col(dcc.Graph(id='combat-scatter'))
            ])
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            # PCA
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-pca')),
                dbc.Col(dcc.Graph(id='combat-pca'))
            ])
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            # clustergram
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-clustergram')),
                dbc.Col(dcc.Graph(id='combat-clustergram'))
            ])
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            # distribution plots
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-distplot-norm')),
                dbc.Col(dcc.Graph(id='combat-distplot-norm'))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='raw-distplot-kde')),
                dbc.Col(dcc.Graph(id='combat-distplot-kde'))
            ]),
            #dbc.Row([
            #    dbc.Col(dcc.Graph(id='combat-prior-dist'))
            #])
        ])
    )
])

tab3_compare = html.Div()

tab4_combat = html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H4("Define your variables for visualization and analysis.",
                className="card-title"),
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
        ])
    ),
    dbc.Card(
        dbc.CardBody([
            html.P(
                "Choose your Combat version:"
            ),
            dbc.RadioItems(options=[
                {"label":"Fortin's neuroCombat","value":1},
                {"label":"Pomponio's neuroHarmonize","value":2},
                {"label":"ENIGMA Combat","value":3}
                ],
                value=1,
                id="combat-run-version"),
            html.Div([html.P(id="combat-version-options"),html.Div([daq.BooleanSwitch(id="switch-eb"),daq.BooleanSwitch(id="switch-parametric"),dcc.Dropdown(id="dropdown-nonlinear")],style={'display':'none'})])
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            html.P(
                "Define the statistical model you'd like to use:"
            ),
            dbc.Input(id="combat-model", placeholder="~x+y", type="text"),
            html.P(),
            html.Div([
                dbc.Button("Get Model Matrix",id="combat-model-submit",n_clicks=0,className="me-md-2",),
            ],className="d-grid gap-2 col-6 mx-auto"),
            html.Div(id='combat-model-output')
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            html.P(
                "Click the 'submit' button to run Combat on your data. "
                "Upon completion, you can return to the 'Plots' tab to "
                "visualize your results in comparison to the raw data."
            ),
            html.Div([
                dbc.Button("Submit",id="combat-run-submit",n_clicks=0,className="me-md-2"),
            ],className="d-grid gap-2 col-6 mx-auto"),
            html.Hr(),
            html.Div(id="combat-run-output") 
        ])
    )
])

tab5_longitudinal = html.Div()

tabs = dbc.Tabs([
    dbc.Tab(tab1_upload,label="Upload"),
    dbc.Tab(tab2_plots,label="Plots"),
    dbc.Tab(tab3_compare,label="Compare Combat Versions"),
    dbc.Tab(tab4_combat,label="Cross-sectional Combat"),
    dbc.Tab(tab5_longitudinal,label="Longitudinal Combat")
])
toast = html.Div([
    dmc.Text(id="txt"),
])

footer = html.Div(
    dbc.Container(
        [
            html.Hr(className="my-2"),
            html.P("Created and maintained by Lezlie España"),
        ],
        fluid=True,
    ),
    className="h-10 p-5 bg-light border rounded-3",
)

app.layout = html.Div([
    header,
    steps,
    tabs,
    toast,
    footer,
    dcc.Store(id="stored-data",storage_type="session"),
    dcc.Store(id="stored-combat",storage_type="memory"),
    dcc.Store(id="stored-combat-model",storage_type="memory"),
    dcc.Store(id="combat-stdout",storage_type="memory")
])

if __name__ == '__main__':
    app.run_server(debug=True,port=8050,host='0.0.0.0')