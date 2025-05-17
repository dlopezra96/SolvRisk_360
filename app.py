"""Entry point for the SolvRisk 360 Dash application."""
import os
import dash
import warnings
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
from typing import Any

load_dotenv()
os.environ["R_HOME"] = os.getenv("R_HOME", "")

# SSuppress rpy2 warnings.
warnings.filterwarnings("ignore", module="rpy2.rinterface")

# Create the Dash app in multi-page mode
app = Dash(
    __name__,
    use_pages=True,  # Activate the multi-page system
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "🔥 Fire Risk Dashboard"

# Layout global: Location, Stores y page_container
layout_children: Any = [
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="stored-data", storage_type="memory"),
    dcc.Store(id="algo-output", storage_type="memory"),
    html.Div(id="algo-status"),
    dash.page_container,
]
app.layout = html.Div(children=layout_children)

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=True)
