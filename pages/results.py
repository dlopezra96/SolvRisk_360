import io
import os
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import math
import dash_leaflet as dl
from modules.kpi_utils_prev import compute_global_kpis
from pages.home import EMOJIS


# -------------------------------------------------------------------
# 1) Register this page at path "/results"
# -------------------------------------------------------------------
dash.register_page(__name__, path="/results", name="Results")

# -------------------------------------------------------------------
# 2) Helper: formatear números al estilo español
# -------------------------------------------------------------------
def format_number_es(value, ndigits=2, use_comma=True):
    """
    Si use_comma=True, formatea en estilo español:
      - Separador de miles: punto
      - Separador decimal: coma
    Si use_comma=False, devuelve un simple formato
      con punto decimal y ndigits decimales.
    """
    try:
        v = float(value)
    except:
        return str(value)
    if not use_comma:
        return f"{v:.{ndigits}f}"
    s = f"{v:,.{ndigits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

# -------------------------------------------------------------------
# 3) Layout: sección de resultados con meta-info + KPI-cards
# -------------------------------------------------------------------
layout = html.Div([
    html.Div(
        html.Img(src="/assets/dlr_logo_white.png", className="logo-top-right"),
        className="logo-wrapper"
    ),
    
    html.H1("📈 Analysis Results", className="dashboard-title"), # PONER EN EL CENTRO
    html.Div(id="results-meta", className="results-meta", style={"marginBottom": "1rem"}),
    html.Div(
        [
            html.Button("⬇️ Download CSV", id="download-csv-btn", className="btn btn-success me-3"),
            html.Button("⬇️ Download PDF report", id="download-pdf-btn", className="btn btn-outline-secondary", 
                        disabled=True),
            dcc.Download(id="download-csv") # solo csv operativo
        ],
        style={
            "display": "flex",
            "justifyContent": "center",            
            "gap": "1rem",                         
            "margin": "1.5rem 0",                  
            "padding": "1rem 2rem",                
            "borderRadius": "0.75rem",               
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"  
        }
    ),    
    html.Div(id="comparisons-table", style={"marginBottom": "1rem"}), 
    html.H2("🎯 Concentration Results", className="section-title"),
    html.Div(id="results-cards", className="kpi-container"),
    html.H2("📊 Global KPIs Results", className="section-title"),
    html.Div(id="global-kpis-cards", className="kpi-container"),  
    html.H2("🗺️ Policy Map Results", className="section-map"),
    html.Div(id="results-map", style={"height": "500px", "marginTop": "2rem"}),  
    html.Footer(
        dcc.Link(
            html.Button("🏠︎ Back to Dashboard", className="btn btn-outline-secondary"),
            href="/"
        ),
        style={                           
            "textAlign": "center",
            "padding": "2rem 0",
            "borderTop": "1px solid #444",
            "marginTop": "2rem"
        }
    )
], className="main-wrapper")

# -------------------------------------------------------------------
# 4) Callback: lee los datos de algo-output y construye las cards
# -------------------------------------------------------------------
@callback(
    [
        Output("results-meta", "children"),
        Output("comparisons-table", "children"),
        Output("results-cards", "children"),
        Output("global-kpis-cards", "children"),
        Output("results-map", "children"),
    ],
    Input("algo-output", "data"),
    State("stored-data", "data")
)

def display_results(algo_output, stored_data):
    # 1) Validation
    if not algo_output or not isinstance(algo_output, dict):
        meta = html.Div("⚠️ No results available.", className="text-muted")
        return meta, None, [], [], None
    
    # 2) Reconstruct original Dataframe
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient = "split")

    # 3) Extract algorithm values
    s   = algo_output.get("sum", 0)
    lat = algo_output.get("lat", None)
    lon = algo_output.get("lon", None)
    cnt = algo_output.get("count", None)
    exec_time = algo_output.get("execution_time", 0.0)
    algo_time = algo_output.get("algorithm_time", 0.0)

    # 4) Calculate totals & percentatge
    total_policies = len(df)
    total_sum      = df["insured_sum"].sum()
    pct_policies   = cnt / total_policies * 100
    pct_sum        = s / total_sum * 100

    # 5) Build meta badges
    file_name = stored_data.get("file_name", "-")
    algorithm = algo_output.get("algorithm", "-")
    params = algo_output.get("parameters", {})
    file_badge = html.Span(
        file_name,
        className="badge bg-primary ms-1 mb-1",
        style={"fontSize": "0.9rem"}
    )
    algo_badge = html.Span(
        algorithm,
        className="badge bg-success ms-1 mb-1",
        style={"fontSize": "0.9rem"}
    )
    param_badges = [
        html.Span(
            f"{k}: {v}",
            className="badge bg-secondary ms-1 mb-1",
            style={"fontSize": "0.85rem"}
        )
        for k, v in params.items()
    ]
    meta = html.Div([
        # 1ª fila: CSV
        html.Div([
            html.Span("🗄️ CSV used:", style={"fontWeight": "bold", "verticalAlign": "middle"}),
            file_badge
        ], style={"display": "flex", "alignItems": "center", "gap": "0.5rem"}),

        # 2ª fila: Algorithm
        html.Div([
            html.Span("⚙️ Algorithm:", style={"fontWeight": "bold", "verticalAlign": "middle"}),
            algo_badge
        ], style={"display": "flex", "alignItems": "center", "gap": "0.5rem", "marginTop": "0.5rem"}),

        # 3ª fila: Parameters
        html.Div(
            [html.Span("🔧 Parameters:", style={"fontWeight": "bold", "verticalAlign": "middle"})] +
            param_badges,
            style={
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "0.25rem",
                "marginTop": "0.5rem"
            }
        )
    ], className="results-meta")

    # 6) Build comparisons table if present
    comparisons = algo_output.get("comparisons")
    if comparisons:
        # cabeceras: añadimos “Center”
        headers = ["Algorithm", "Sum", "# policies", "Center", "Time (s)"]
        thead = html.Thead(html.Tr([html.Th(col) for col in headers]))
        
        # filas: incluimos un td con (lat, lon)
        tbody = html.Tbody([
            html.Tr([
                html.Td(r["algo"]),
                html.Td(format_number_es(r["sum"], ndigits=2)),
                html.Td(str(r["count"])),
                html.Td(
                    f"({format_number_es(r['center'][0], ndigits=6, use_comma=False)}, "
                    f"{format_number_es(r['center'][1], ndigits=6, use_comma=False)})"
                ),
                html.Td(format_number_es(r["time"], ndigits=5, use_comma=False)),
            ])
            for r in comparisons
        ])
        
        table = html.Table([thead, tbody])
        comp_table = html.Div(table, className="table-wrapper")
    else:
        comp_table = None

    # 7) Build KPI cards (Concetrarion results)
    conc_cards = [
        html.Div([html.H4("🗂️ Policies"), 
                  html.P(format_number_es(cnt, ndigits=0))], className="kpi-card"),
        html.Div([html.H4("📊 % of policies"), 
                  html.P(format_number_es(pct_policies, ndigits=2) + "%")], className="kpi-card"),
        html.Div([html.H4("💰 Total sum (€)"), 
                  html.P(format_number_es(s, ndigits=2))], className="kpi-card"),
        html.Div([html.H4("📊 % insured sum"), 
                  html.P(format_number_es(pct_sum, ndigits=2) + "%")], className="kpi-card"),
        html.Div([html.H4("📍 Center latitude"), 
                  html.P(format_number_es(lat, ndigits=6, use_comma=False))], className="kpi-card"),
        html.Div([html.H4("📍 Center longitude"), 
                  html.P(format_number_es(lon, ndigits=6, use_comma=False))], className="kpi-card"),
        html.Div([html.H4("⏱️ Total time (s)"), 
                  html.P(f"{format_number_es(exec_time, ndigits=6, use_comma=False)}")], className="kpi-card"),
        html.Div([html.H4("⚡ Algorithm time (s)"), 
                  html.P(f"{format_number_es(algo_time, ndigits=6, use_comma=False)}")], className="kpi-card"),
    ]

    # 7) Build Global kpis
    idx = algo_output.get("indices", [])
    inside = df.iloc[idx]

    inside_kpis      = compute_global_kpis(inside)
    inside_kpis.pop("Total Policies", None)   
    global_kpi_cards = [
        *[
            html.Div([
                html.H4(f"{EMOJIS[key]} {key}"),
                html.P(format_number_es(value))
            ], className="kpi-card")
            for key, value in inside_kpis.items()
        ],
        html.Div([
            html.H4("—"), 
            html.P("")      
        ], className="kpi-card"),
    ]
    
    # 8) BUILD MAP: show ONLY inside points + red circle   
    tile_layer = dl.TileLayer(
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution="© CartoDB"
    )

    markers = [
        dl.Marker(
            position=(row["lat"], row["lon"]),
            children=dl.Popup(f"Policy: {row['n_policy']} | {row['insured_sum']}€")
        )
        for _, row in inside.iterrows()
    ]

    R = algo_output.get("parameters", {}).get("radius", 200)
    circle = dl.Circle(center=(lat, lon), radius=R, color="red", weight=2, fill=False)

    # Configure zoom
    dlat = R / 111320
    dlon = R / (111320 * math.cos(math.radians(lat)))
    bounds = [
        [lat - dlat, lon - dlon],
        [lat + dlat, lon + dlon]
    ]

    # Map
    leaflet_map = dl.Map(
        children=[tile_layer, circle, dl.LayerGroup(markers)],
        bounds=bounds,
        boundsOptions={"padding": [50, 50]}, 
        style={"width": "100%", "height": "100%"}
    )

    return meta, comp_table, conc_cards, global_kpi_cards, leaflet_map

# -------------------------------------------------------------------
# 10) Callback para descarga de CSV completo
# -------------------------------------------------------------------
@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("algo-output", "data"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def download_selected_csv(n_clicks, algo_output, stored_data):
    # 1) Rebuild the DataFrame from the selected rows
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient="split")
    idx = algo_output.get("indices", [])
    inside = df.iloc[idx]

    # 2) Generate timestamp and output path
    ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"conc_results_{ts}.csv"
    filepath = os.path.join(reports_dir, filename)

    # 3) Save the CSV to disk
    inside.to_csv(filepath, index=False)

    # 4) Sent the CSV to the client with a generic name
    return dcc.send_data_frame(inside.to_csv, filename, index=False)