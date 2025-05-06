import io
import tempfile
import dash
import pandas as pd
import numpy as np
import time
from dash import html, dcc, callback, Input, Output, State, ALL, dash_table
from dash.exceptions import PreventUpdate
from modules.concentration_algorithms.by_papers import grid_search_r
from modules.concentration_algorithms.new_methods import fast_with_grid, fast_with_multilocal
from modules.spatial_index import haversine_coords, build_balltree, tree_radius

# -------------------------------------------------------------------
# 1) Register this page at path "/select-algorithm"
# -------------------------------------------------------------------
dash.register_page(__name__, path="/select-algorithm", name="Select Algorithm")

# -------------------------------------------------------------------
# 2) Define your algorithms and rich descriptions
# -------------------------------------------------------------------
ALGO_LIST = [
    {
        "algo": "grid_search_r",
        "description": (
            "Calls the CRAN spatialrisk package’s highest_concentration function in R to find the optimal fire-risk "
            "circle: it converts lat/lon to geohashes, sums insured values per hash (with padding), and scans a fine 25m "
            "grid in C++ for the true maximum. Highly accurate but depends on rpy2 and R."
        )
    },
    {
        "algo": "fast_with_grid",
        "description": (
            "Hybridizes an exhaustive discrete search with local grid tuning: step 1 ranks each policy via a BallTree "
            "radius query; step 2 selects the top k seeds; step 3 generates a 25 m grid in a ±200 m square around "
            "each seed and re-computes sums. Pure-Python, yields high precision and great speed on large datasets."
        )
    },
    {
        "algo": "fast_with_multilocal",
        "description": (
            "Hybridizes an exhaustive discrete search with a micro multi-start pattern search: step 1 ranks each policy via "
            "a BallTree radius query; step 2 selects the top k seeds; step 3 runs a short pattern-search around each seed—"
            "moving in cardinal directions with adaptive step sizes until a 25 m threshold; step 4 picks the best "
            "continuous centre across all starts. Pure-Python, balances blazing speed with fine-grained accuracy on large portfolios."
        )
    },
    {
        "algo": "all",
        "description": (
            "Runs all three algorithms above and picks the cluster with the highest insured sum "
            "(ties broken by fastest execution time)."
        )
    }
]

# -------------------------------------------------------------------
# 3) Define parameter inputs per algorithm
# -------------------------------------------------------------------
ALGO_PARAMS = {
    "grid_search_r": [
        {"id": "radius",        "label": "Radius (m)",       "type": "number", "min": 1, "step": 1, "value": 200},
        {"id": "grid_distance", "label": "Grid step (m)",    "type": "number", "min": 1, "step": 1, "value": 25},
    ],
    "fast_with_grid": [
        {"id": "radius",         "label": "Radius (m)",      "type": "number",   "min": 1, "step": 1, "value": 200},
        {"id": "grid_distance",  "label": "Grid step (m)",   "type": "number",   "min": 1, "step": 1, "value": 25},
        {"id": "top_k",          "label": "Top K anchors",   "type": "number",   "min": 1, "step": 1, "value": 10},
        {"id": "metric",         "label": "Distance metric", "type": "dropdown","options": [
            {"label": "Haversine", "value": "haversine"},
            {"label": "Euclidean", "value": "euclidean"},
        ], "value": "haversine"}
    ],
    "fast_with_multilocal": [
        {"id": "radius",        "label": "Radius (m)",       "type": "number", "min": 1, "step": 1, "value": 200},
        {"id": "top_k",         "label": "Top K anchors",    "type": "number", "min": 1, "step": 1, "value": 10},
        {"id": "delta0",        "label": "Initial step (m)", "type": "number", "min": 1, "step": 1, "value": 400},
        {"id": "delta_min",     "label": "Min step (m)",     "type": "number", "min": 1, "step": 1, "value": 25},
        {"id": "i_min",         "label": "Max iters (i_min)","type": "number", "min": 1, "step": 1, "value": 100},
        {"id": "metric",        "label": "Distance metric",  "type": "dropdown","options": [
            {"label": "Haversine", "value": "haversine"},
            {"label": "Euclidean", "value": "euclidean"},
        ], "value": "haversine"}
    ],
    "all": [
        {"id": "radius",        "label": "Radius (m)",       "type": "number",   "min": 1, "step": 1, "value": 200},
        {"id": "grid_distance", "label": "Grid step (m)",    "type": "number",   "min": 1, "step": 1, "value": 25},
        {"id": "top_k",         "label": "Top K anchors",    "type": "number",   "min": 1, "step": 1, "value": 10},
        {"id": "i_min",         "label": "Max iters",        "type": "number",   "min": 1, "step": 1, "value": 100},
        {"id": "delta0",        "label": "Initial step (m)", "type": "number",   "min": 1, "step": 1, "value": 400},
        {"id": "delta_min",     "label": "Min step (m)",     "type": "number",   "min": 1, "step": 1, "value": 25},
        {"id": "metric",        "label": "Distance metric",  "type": "dropdown","options": [
            {"label": "Haversine", "value": "haversine"},
            {"label": "Euclidean", "value": "euclidean"},
        ], "value": "haversine"}
    ]
}

# -------------------------------------------------------------------
# 4) Layout: DataTable + dynamic params + Run + status + Back link
# -------------------------------------------------------------------
layout = html.Div([
    html.Div(
        html.Img(src="/assets/dlr_logo_white.png", className="logo-top-right"),
        className="logo-wrapper"
    ),
    html.H1("⚙️ Select an algorithm to run", className="dashboard-title"),

    html.Div(id="debug-json", style={
        "marginBottom": "1rem", "padding": "0.5rem",
        "backgroundColor": "#222", "color": "#0f0",
        "whiteSpace": "pre-wrap", "fontSize": "0.8rem",
        "fontFamily": "monospace"
    }),

    dash_table.DataTable(
        id="algo-table",
        columns=[
            {"name": "Algorithm",   "id": "algo"},
            {"name": "Description", "id": "description"}
        ],
        data=ALGO_LIST,
        row_selectable="single",
        selected_rows=[],
        style_table={"width": "100%", "marginBottom": "1rem"},
        style_cell_conditional=[
            {"if": {"column_id": "algo"},
             "width": "20%", "fontSize": "0.9rem", "fontFamily": "monospace"},
            {"if": {"column_id": "description"},
             "width": "80%", "fontSize": "0.7rem", "lineHeight": "1.4", "whiteSpace": "pre-wrap"}
        ],
        style_cell={"textAlign": "left", "padding": "10px", "height": "auto"},
        style_header={
            "backgroundColor": "#1a1d22",
            "color": "#ffffff",
            "fontWeight": "bold",
            "border": "none",
            "fontSize": "1.1rem"
        },
        style_data={"backgroundColor": "#1a1d22", "color": "#ffffff", "border": "none"},
        style_data_conditional=[
            {"if": {"state": "selected"},
             "backgroundColor": "#264653", "color": "#ffffff", "border": "none"}
        ]
    ),

    html.Div(id="algo-params-container", style={"margin": "1rem 0"}),

    html.Button("Run analysis", id="run-algo-btn", n_clicks=0, className="btn btn-success", disabled=True),

    html.Div(
        html.Progress(id="progress-bar", max="100"),
        id = "progress-container",
        style={'display': 'none', "margin": "1rem 0"}
    ),

    html.Div(id="algo-status", style={"marginTop": "1rem"}),
], className="main-wrapper")

# -------------------------------------------------------------------
# 5) Render the parameter inputs
# -------------------------------------------------------------------
@callback(
    Output("algo-params-container", "children"),
    Input("algo-table", "selected_rows"),
    State("algo-table", "data")
)

def render_algo_params(selected_rows, table_data):
    if not selected_rows:
        return []

    algo_key = table_data[selected_rows[0]]["algo"]
    params = ALGO_PARAMS.get(algo_key, [])
    if not params:
        return html.Div("No extra parameters for this algorithm.", className="text-muted")

    controls = []
    for p in params:
        inp_id = {"type": "param-input", "index": p["id"]}
        label = html.Label(p["label"], style={"marginRight": "0.3rem"})
        if p["type"] == "dropdown":
            control = dcc.Dropdown(
                id=inp_id,
                options=p["options"],
                value=p["value"],
                clearable=False,
                className="param-dropdown",
                style={"width": "8rem"}
            )
        else:
            control = dcc.Input(
                id=inp_id,
                type="number",
                min=p["min"],
                step=p["step"],
                value=p["value"],
                style={"width": "6rem"}
            )
        controls.append(
            html.Div([label, control],
                     style={
                         "display": "flex",
                         "flexDirection": "column",
                         "alignItems": "flex-start",
                         "marginRight": "1rem"
                     })
        )

    return html.Div(controls, style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
                                     "justifyContent": "flex-start", "gap": "1rem"})

# ------------------------------------------------------------------------------
# 6) Enable the Run button only when exactly one row is selected and data loaded
# ------------------------------------------------------------------------------
@callback(
    Output("run-algo-btn", "disabled"),
    [Input("algo-table", "selected_rows"), Input("stored-data", "data")]
    )

def toggle_run_button(selected_rows, stored_json):
    if stored_json is None:
        return True
    return len(selected_rows) != 1

# -------------------------------------------------------------------
# 7) Debug: show stored-data status
# -------------------------------------------------------------------
@callback(
    Output("debug-json", "children"),
    Input("stored-data", "data")
)
def debug_json(data):
    if data is None or not isinstance(data, dict) or "df_json" not in data:
        return "⚠️ stored-data is None or invalid (no JSON received)"

# -------------------------------------------------------------------
# 8) Show progress-bar when clicking Run analysis
# -------------------------------------------------------------------
@callback(
    Output("progress-container", "style"),
    Input("run-algo-btn", "n_clicks"),
    prevent_initial_call=True
)

def toggle_progress_container(n):
    return {'display':'block','margin':'1rem 0'}

# -------------------------------------------------------------------
# 9) Callback “Run analysis”
# -------------------------------------------------------------------
@callback(
    [
        Output("algo-output", "data"),
        Output("algo-status", "children"),
        Output("progress-bar", "value"),
    ],
    Input("run-algo-btn", "n_clicks"),
    [
        State("algo-table", "selected_rows"),
        State("algo-table", "data"),
        State("stored-data", "data"),
        State({"type": "param-input", "index": ALL}, "value")
    ],
    prevent_initial_call=True
)

def run_analysis(n_clicks, selected_rows, table_data, stored_data, param_values):
    if stored_data is None or not selected_rows:
        raise PreventUpdate

    # 1) Deserialize DataFrame from JSON
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient="split")
    algo_key = table_data[selected_rows[0]]["algo"]
    
    # 2) Execute
    t0 = time.time()
    # --- individual mode ---
    if algo_key == "grid_search_r":
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        cnt, s, (lat, lon), algo_time = grid_search_r(tmp.name, *param_values)
        coords = haversine_coords(df["lat"].values, df["lon"].values)
        tree = build_balltree(coords, metric='haversine')
        r_tree = tree_radius(param_values[0], metric='haversine')
        center_rad = np.array([[np.deg2rad(lat), np.deg2rad(lon)]])
        idx = list(tree.query_radius(center_rad, r_tree)[0])

    elif algo_key == "fast_with_grid":
        t_alg = time.time()
        idx, s, (lat, lon) = fast_with_grid(df, *param_values)
        algo_time = time.time() - t_alg
        cnt = len(idx)

    elif algo_key == "fast_with_multilocal":
        t_alg = time.time()
        idx, s, (lat, lon) = fast_with_multilocal(df, *param_values)
        algo_time = time.time() - t_alg
        cnt = len(idx)

    # --- mode "all" ---
    else:
        # run all three, pick best
        results = []
        for key, fn in [("grid_search_r", grid_search_r),
                        ("fast_with_grid", fast_with_grid),
                        ("fast_with_multilocal", fast_with_multilocal)]:
            #start_i = time.time()
            if key == "grid_search_r":
                tmp2 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                df.to_csv(tmp2.name, index=False)
                t_sub = time.time()
                # only radius & grid_distance
                cnt_i, s_i, center_i, algo_t_i = fn(tmp2.name, param_values[0], param_values[1])
                sub_elapsed = time.time() - t_sub
                coords = haversine_coords(df["lat"].values, df["lon"].values)
                tree = build_balltree(coords, metric='haversine')
                r_tree = tree_radius(param_values[0], metric='haversine')
                center_rad = np.array([[np.deg2rad(center_i[0]), np.deg2rad(center_i[1])]])
                idx_i = list(tree.query_radius(center_rad, r_tree)[0])

            elif key == "fast_with_grid":
                t_sub = time.time()
                # radius, grid_distance, top_k, metric at index 3
                idx_i, s_i, center_i = fn(df,
                       param_values[0], # radius
                       param_values[1], # grid_distance
                       param_values[2], # top_k
                       param_values[6]  # metric
                       )
                algo_t_i = time.time() - t_sub
                sub_elapsed = algo_t_i
                cnt_i = len(idx_i)
            else:
                # radius, top_k, i_min, delta0, delta_min, metric
                t_sub = time.time()
                idx_i, s_i, center_i = fn(df,
                       param_values[0], # radius
                       param_values[2], # top_k
                       param_values[4], # delta0
                       param_values[5], # deltamin
                       param_values[3], # i_min
                       param_values[6]) # metric
                algo_t_i = time.time() - t_sub
                sub_elapsed = algo_t_i
                cnt_i = len(idx_i)
                   
            results.append({
                "algo":    key,
                "sum":     s_i,
                "count":   cnt_i,
                "center":  center_i,
                "time":    sub_elapsed,
                "indices": idx_i
            })

        best = sorted(results, key=lambda x: (x["sum"], -x["time"]), reverse=True)[0]

        s, cnt, (lat, lon), algo_time, chosen_key, idx = (
            best["sum"],
            best["count"],
            best["center"],
            best["time"],
            best["algo"],
            best["indices"]
            )

        algo_key = f"all_{chosen_key}"       

    total_time = time.time() - t0

    # 5) Pack the result
    base_key = algo_key
    if algo_key.startswith("all_"):
        base_key = "all"   
    param_names = [p["id"] for p in ALGO_PARAMS[base_key]]
    params_dict = dict(zip(param_names, param_values))
    
    algo_output = {
        "sum":             s,
        "lat":             lat,
        "lon":             lon,
        "count":           cnt,
        "indices":         idx,
        "algorithm":       algo_key,
        "execution_time":  total_time,
        "algorithm_time":  algo_time,
        "parameters":      params_dict
    }

    if algo_key.startswith("all_"):
        algo_output["comparisons"] = results

    # 6) Final status message
    status = html.Div([
        html.Span(f"✅ Done in {total_time:.2f}s"),
        dcc.Link("📊 See results", href="/results",
                 className="btn btn-primary",
                 style={"marginLeft": "1rem"})
    ], style={"display": "flex", "alignItems": "center"})

    return algo_output, status, "100"