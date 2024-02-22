import dash
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from dash_extensions.enrich import html, DashProxy, LogTransform, dcc 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_icon(icon):
    return DashIconify(icon=icon, height=16)

app = DashProxy(__name__,external_stylesheets=[dbc.themes.FLATLY,dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True,transforms=[LogTransform()], prevent_initial_callbacks=True,use_pages=True)

navbar = dbc.NavbarSimple(
       children=[
        dbc.NavItem(dbc.NavLink(
            "Compare Combat",
            href=dash.page_registry['pages.comparison']['path'],
            class_name="text-white"
        )),
        dbc.NavItem(dbc.NavLink(
            "Cross-Sectional",
            href=dash.page_registry['pages.cross_sectional']['path'],
            class_name="text-white"
        )),
        dbc.NavItem(dbc.NavLink(
            "Longitudinal",
            href=dash.page_registry['pages.longitudinal']['path'],
            class_name="text-white"
        )),
    ],
    brand="Data Harmonization",
    brand_href=dash.page_registry['pages.home']['path'],
    fluid=True,
    color="primary",
    dark=True,
)

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
    navbar,
    dash.page_container,
    footer,
])


if __name__ == '__main__':
    app.run_server(debug=True,port=8050,host='0.0.0.0')