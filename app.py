import os
import dash
import warnings
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()
os.environ["R_HOME"] = os.getenv("R_HOME")

# Suprimir los warnings de rpy2
warnings.filterwarnings("ignore", module="rpy2.rinterface")

# Creamos la app Dash en modo multipágina
app = Dash(
    __name__,
    use_pages=True,              # activa el sistema multipágina
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "🔥 Fire Risk Dashboard"

# Layout global: Location, Stores y page_container
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='stored-data', storage_type='memory'),
    dcc.Store(id='algo-output', storage_type='memory'),
    html.Div(id='algo-status'),
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=True)
