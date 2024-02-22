import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import html, dcc

dash.register_page(
    __name__,
    path='/home',
    title='Data Harmonization Dashboard',
    name='Data Harmonization Dashboard'
)

layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H4(
                "This web app is designed to facilitate the "
                "analysis and visualization of data that may "
                "be affected by batch or site differences. "
                "You can upload your dataset, view various plots, "
                "run multiple versions of Combat (depending on your data), "
                "and compare results. See the steps below to get started!"
            )
        ])
    ),
    html.Hr(),
    dbc.Card(
        dbc.CardBody([
            html.H3("Instructions for use: "),
            html.P(
                "If you know what version of Combat you'd like to run, "
                "choose the appropriate page from the menu to get started. "
                "Otherwise you can choose to 'Compare Combat' page to "
                "explore your options. "
                "Regardless of the version you choose, the following steps "
                "will apply: "
            ),
            dcc.Markdown('''
                * Upload your data (preferrably in CSV format)
                * Choose your variables:
                    * Variables of interest - variables you may
                         want to adjust via Combat
                    * Batch/Site variable
                    * Any additional variables:
                        * ID, Time variables for longitudinal Combat
                        * "Conditional" variables or Covariates to consider
                * Choose your Combat variation and any additional options
                    * Below you'll find more information about each version
                * Define your statistical model
                * Submit and run Combat
                * View and download the data
                * Head to the "Plots" page to visualize your data and the effects of Combat
            ''')
        ])
    ),
    dbc.Card(
        dbc.CardBody([
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
                ]),
                dbc.Col([
                    dcc.Link("Beer's LongCombat",href="https://github.com/jcbeer/longCombat")
                ])
            ]),
        ])
    )
    
])