import os
import io
import base64
import dash
from dash import html, dcc, Input, Output, State, callback, callback_context
import pandas as pd
from modules.kpi_utils_prev import compute_global_kpis, compute_kpis_by_zone
import dash_leaflet as dl
import dash_leaflet.express as dlx

# -------------------------------------------------------------------
# 1) register this script as the home page at path "/"
# -------------------------------------------------------------------
dash.register_page(__name__, path="/", name="Home")

# -------------------------------------------------------------------
# 2) constants and helper functions
# -------------------------------------------------------------------
DATA_FOLDER = "data/"
available_files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(".csv")
]

def format_number_es(value):
    """
    format a number using Spanish locale conventions:
    - thousands separator: dot
    - decimal separator: comma
    """
    try:
        return (
            f"{float(value):,.2f}"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    except:
        return str(value)

# emoji icons for each KPI
EMOJIS = {
    "Total Policies":    "🗂️",
    "Minimum (€)":       "📉",
    "1st Quartile (€)":  "¼️",
    "Median (€)":        "➗",
    "3rd Quartile (€)":  "¾️",
    "Mean (€)":          "⚖️",
    "Maximum (€)":       "📈",
    "St.Desv. (€)":      "📐"
}

# -------------------------------------------------------------------
# 3) page layout: logo, title, selection controls, hidden sections
# -------------------------------------------------------------------
layout = html.Div([
    # logo in top-right corner
    html.Div(
        html.Img(src="/assets/dlr_logo_white.png", className="logo-top-right"),
        className="logo-wrapper"
    ),

    # main title
    html.H1("🔥 Fire Risk Dashboard", className="dashboard-title"),

    # data selection row
    html.Div([
        # dropdown selector
        html.Div([
            html.Label("📂 Select dataset:", className="dropdown-label"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[{"label": f, "value": f} for f in available_files],
                placeholder="Select a sample CSV",
                clearable=True
            )
        ], className="selection-box"),
        # file upload component
        html.Div([
            html.Label("🗂️ Or upload a CSV file:", className="dropdown-label"),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "📁 Drag and drop or ",
                    html.Span("select a CSV file",
                              style={"textDecoration": "underline", "cursor": "pointer"})
                ]),
                multiple=False
            )
        ], className="box-container")
    ], className="upload-row"),

    # hidden button to navigate to algorithm selection
    html.Div(
        dcc.Link("⚙️ Select algorithm and execute",
                 href="/select-algorithm", className="btn btn-primary"),
        id="select-button-div",
        style={"display": "none", "textAlign": "center", "margin": "1rem 0"}
    ),

    # area to show which file is being used
    html.Div(id="file-info", style={"textAlign": "center", "marginBottom": "20px"}),

    # global kpis section (hidden initially)
    html.Div([
        html.H2("📊 Global KPIs"),
        html.Div(id="kpi-cards", className="kpi-container")
    ], id="global-kpi-section", style={"display": "none"}),

    # kpis by zone section (hidden initially)
    html.Div([
        html.H2("📍 KPIs by Zone"),
        html.Div(id="zone-kpi-table", className="table-wrapper")
    ], id="zone-kpi-section", style={"display": "none"}),

    # map section (hidden initially)
    html.Div([
        html.H2("🗺️ Policy Map", className="section-title"),
        html.Div(id="map-container", style={"height": "500px", "marginTop": "2rem"})
    ], id="map-section", style={"display": "none"}),
], className="main-wrapper")

# -------------------------------------------------------------------
# 4) callback: load csv, compute kpis & map, store data, show/hide sections
# -------------------------------------------------------------------
@callback(
    [
        Output("map-container",      "children"),
        Output("kpi-cards",          "children"),
        Output("zone-kpi-table",     "children"),
        Output("file-info",          "children"),
        Output("stored-data",        "data"),     # store dataframe json
        Output("select-button-div",  "style"),    # show/hide select button
        Output("global-kpi-section", "style"),    # show/hide global kpis
        Output("zone-kpi-section",   "style"),    # show/hide zone kpis
        Output("map-section",        "style"),    # show/hide map
    ],
    [
        Input("dataset-dropdown",    "value"),
        Input("upload-data",         "contents")
    ],
    [State("upload-data", "filename")],
    prevent_initial_call=True
)
def update_dashboard(selected_file, uploaded_contents, uploaded_filename):
    ctx = callback_context
    trigger = (ctx.triggered[0]["prop_id"].split(".")[0]
               if ctx.triggered else None)

    # if no data yet: hide everything
    if not ((trigger == "upload-data" and uploaded_contents and uploaded_filename)
            or selected_file):
        hidden = {"display": "none"}
        return (
            [],      # map-graph
            [],      # kpi-cards
            [],      # zone-kpi-table
            "",      # file-info
            dash.no_update,  # stored-data
            hidden,  # select-button-div
            hidden,  # global-kpi-section
            hidden,  # zone-kpi-section
            hidden   # map-section
        )

    # load dataframe from upload or dropdown
    if trigger == "upload-data":
        _, content_string = uploaded_contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        file_msg = f"📄 Using uploaded file: {uploaded_filename}"
    else:
        df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
        file_msg = f"📄 Using file: {selected_file}"

    # create map figure
    tile_layer = dl.TileLayer(
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution="© CartoDB"
    )

    geojson = dlx.dicts_to_geojson(
        df.to_dict("records"),
        lon="lon", lat="lat")

    leaflet_map = dl.Map(
        children=[
            tile_layer,
            dl.GeoJSON(
                data=geojson,
                cluster=True,                                       # ENABLES CLUSTERING
                superClusterOptions={"radius": 50, "maxZoom": 18},  # OPTIONAL: TUNE CLUSTERING
                zoomToBoundsOnClick=True                            # OPTIONAL: ZOOM INTO CLUSTER ON CLICK
            )
        ],
        center=(df["lat"].mean(), df["lon"].mean()),
        zoom=5.5,
        style={"width": "100%", "height": "100%"}
    )

    # compute global kpis
    global_kpis = compute_global_kpis(df)
    cards = [
        html.Div([html.H4(f"{EMOJIS.get(key, '')} {key}"),
                  html.P(format_number_es(value))],
                 className="kpi-card")
        for key, value in global_kpis.items()
    ]

    # compute kpis by zone
    zone_df = compute_kpis_by_zone(df)
    headers = ["📍" if col == "zone" else EMOJIS.get(col, "")
               for col in zone_df.columns]
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers])),
        html.Tbody([
            html.Tr([html.Td(
                format_number_es(val)
                if isinstance(val, (int, float)) else val
            ) for val in row])
            for row in zone_df.values
        ])
    ])

    # prepare data for storage
    store_payload = {
        "df_json": df.to_json(date_format="iso", orient="split"),
        "file_name": uploaded_filename or selected_file
    }

    # styles for visible sections
    visible_button  = {"display": "block", "textAlign": "center", "margin": "1rem 0"}
    visible_section = {"display": "block"}

    return (
        leaflet_map,    # map-graph
        cards,          # kpi-cards
        table,          # zone-kpi-table
        file_msg,       # file-info
        store_payload,  # stored-data
        visible_button, # select-button-div
        visible_section,# global-kpi-section
        visible_section,# zone-kpi-section
        visible_section # map-section
    )
