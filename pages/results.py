"""Results page for the SolvRisk dash app."""
import io
import os
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import math
import dash_leaflet as dl
from modules.kpi_utils_prev import compute_global_kpis, compute_kpis_by_zone
from modules.pdf_utils import draw_header, capture_dash_map
from pages.home import EMOJIS
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import folium
import tempfile
from typing import Any, Union, Dict, Optional, Tuple, List

# -------------------------------------------------------------------
# 1) Register this page at path "/results"
# -------------------------------------------------------------------
register_page: Any = dash.register_page
register_page(__name__, path="/results", name="Results")


# -------------------------------------------------------------------
# 2) Helper: format numbers in Spanish style.
# -------------------------------------------------------------------
def format_number_es(value: Union[str, int, float],
                     ndigits: int = 2,
                     use_comma: bool = True) -> str:
    """
    Format a number according to Spanish locale conventions.

    Converts the input to float and formats with a dot as
    thousands separator and a comma as decimal separator
    when `use_comma` is True; otherwise returns a simple
    decimal format with a point.

    Parameters:
    - value (Union[str, int, float]): The value to format.
    - ndigits (int): Number of decimal places.
    - use_comma (bool): If True, uses comma as decimal
      separator and dot for thousands; if False, uses
      standard point decimal.

    Returns:
    - str: The formatted number string, or the original
      value as string if conversion fails.
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return str(value)
    if not use_comma:
        return f"{v:.{ndigits}f}"
    s = f"{v:,.{ndigits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# -------------------------------------------------------------------
# 3) Layout: sección de resultados con meta-info + KPI-cards
# -------------------------------------------------------------------
layout = html.Div(
    [
        html.Div(
            html.Img(src="/assets/dlr_logo_white.png",
                     className="logo-top-right"),
            className="logo-wrapper",
        ),
        html.H1(
            "📈 Analysis Results", className="dashboard-title"
        ),
        html.Div(
            id="results-meta", className="results-meta",
            style={"marginBottom": "1rem"}
        ),
        html.Div(
            [
                html.Button(
                    "⬇️ Download CSV",
                    id="download-csv-btn",
                    className="btn btn-success me-3",
                ),
                html.Button(
                    "⬇️ Download PDF report",
                    id="download-pdf-btn",
                    className="btn btn-success me-3",
                ),
                dcc.Download(id="download-csv"),
                dcc.Download(id="download-pdf"),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "1rem",
                "margin": "1.5rem 0",
                "padding": "1rem 2rem",
                "borderRadius": "0.75rem",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            },
        ),
        html.Div(id="comparisons-table", style={"marginBottom": "1rem"}),
        html.H2("🎯 Concentration Results", className="section-title"),
        html.Div(id="results-cards", className="kpi-container"),
        html.H2("📊 Global KPIs Results", className="section-title"),
        html.Div(id="global-kpis-cards", className="kpi-container"),
        html.H2("🗺️ Policy Map Results", className="section-map"),
        html.Div(id="results-map",
                 style={"height": "500px", "marginTop": "2rem"}),
        html.Footer(
            dcc.Link(
                html.Button(
                    "🏠︎ Back to Dashboard",
                    className="btn btn-outline-secondary"
                ),
                href="/",
            ),
            style={
                "textAlign": "center",
                "padding": "2rem 0",
                "borderTop": "1px solid #444",
                "marginTop": "2rem",
            },
        ),
    ],
    className="main-wrapper",
)


# -------------------------------------------------------------------
# 4) Callback: read the data from algo-output and build the cards
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
    State("stored-data", "data"),
)
def display_results(algo_output: Any,
                    stored_data: Dict[str, Any]
                    ) -> Tuple[
                        html.Div,  # meta
                        Optional[html.Div],  # comp_table
                        List[html.Div],  # conc_cards
                        List[html.Div],  # global_kpi_cards
                        dl.Map]:  # deaflet_map
    """
    Render algorithm results into UI components.

    Reads the stored dataset and algorithm output to build:
    - a metadata summary,
    - an optional comparisons table,
    - KPI cards for the concentration results,
    - KPI cards for the global results of the cluster,
    - and a map highlighting the selected cluster.

    Parameters:
    - algo_output (Any): Dictionary containing algorithm results,
      including 'indices', 'sum', 'center', etc.
    - stored_data (Dict[str, Any]): Stored payload with keys
      'df_json' and 'file_name'.

    Returns:
    - meta (html.Div): A div with badges for file, algorithm,
      and parameters.
    - comp_table (Optional[html.Div]): A div wrapping the comparisons
      table, or None if no comparisons.
    - conc_cards (List[html.Div]): KPI cards for concentration results.
    - global_kpi_cards (List[html.Div]): KPI cards for the cluster's
      global KPIs.
    - leaflet_map (dl.Map): A Dash Leaflet map showing the selected
      points and circle.
    """
    # 1) Validation
    if not algo_output or not isinstance(algo_output, dict):
        meta = html.Div("⚠️ No results available.", className="text-muted")
        return meta, None, [], [], None

    # 2) Reconstruct original Dataframe
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient="split")

    # 3) Extract algorithm values
    s = algo_output.get("sum", 0)
    lat = algo_output.get("lat", None)
    lon = algo_output.get("lon", None)
    cnt = algo_output.get("count", 0)
    exec_time = algo_output.get("execution_time", 0.0)
    algo_time = algo_output.get("algorithm_time", 0.0)

    # 4) Calculate totals & percentatge
    total_policies = len(df)
    total_sum = df["insured_sum"].sum()
    pct_policies = cnt / total_policies * 100
    pct_sum = s / total_sum * 100

    # 5) Build meta badges
    file_name = stored_data.get("file_name", "-")
    algorithm = algo_output.get("algorithm", "-")
    params = algo_output.get("parameters", {})
    file_badge = html.Span(
        file_name, className="badge bg-primary ms-1 mb-1",
        style={"fontSize": "0.9rem"}
    )
    algo_badge = html.Span(
        algorithm, className="badge bg-success ms-1 mb-1",
        style={"fontSize": "0.9rem"}
    )
    param_badges = [
        html.Span(
            f"{k}: {v}",
            className="badge bg-secondary ms-1 mb-1",
            style={"fontSize": "0.85rem"},
        )
        for k, v in params.items()
    ]
    meta = html.Div(
        [
            # CSV
            html.Div(
                [
                    html.Span(
                        "🗄️ CSV used:",
                        style={"fontWeight": "bold",
                               "verticalAlign": "middle"},
                    ),
                    file_badge,
                ],
                style={"display": "flex", "alignItems": "center",
                       "gap": "0.5rem"},
            ),
            # Algorithm
            html.Div(
                [
                    html.Span(
                        "⚙️ Algorithm:",
                        style={"fontWeight": "bold",
                               "verticalAlign": "middle"},
                    ),
                    algo_badge,
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "0.5rem",
                    "marginTop": "0.5rem",
                },
            ),
            # Parameters
            html.Div(
                [
                    html.Span(
                        "🔧 Parameters:",
                        style={"fontWeight": "bold",
                               "verticalAlign": "middle"},
                    )
                ]
                + param_badges,
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "gap": "0.25rem",
                    "marginTop": "0.5rem",
                },
            ),
        ],
        className="results-meta",
    )

    # 6) Build comparisons table if present
    comparisons = algo_output.get("comparisons")
    if comparisons:
        headers = ["Algorithm", "Sum", "# policies", "Center", "Time (s)"]
        thead = html.Thead(html.Tr([html.Th(col) for col in headers]))
        tbody = html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(r["algo"]),
                        html.Td(format_number_es(r["sum"], ndigits=2)),
                        html.Td(str(r["count"])),
                        html.Td(
                            f"({format_number_es(r['center'][0], ndigits=6,
                                                 use_comma=False)}, "
                            f"{format_number_es(r['center'][1], ndigits=6,
                                                use_comma=False)})"
                        ),
                        html.Td(
                            format_number_es(r["time"], ndigits=5,
                                             use_comma=False)
                        ),
                    ]
                )
                for r in comparisons
            ]
        )

        table = html.Table([thead, tbody])
        comp_table = html.Div(table, className="table-wrapper")
    else:
        comp_table = None

    # 7) Build KPI cards (concetration results)
    conc_cards = [
        html.Div(
            [html.H4("🗂️ Policies"), html.P(format_number_es(cnt, ndigits=0))],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("📊 % of policies"),
                html.P(format_number_es(pct_policies, ndigits=2) + "%"),
            ],
            className="kpi-card",
        ),
        html.Div(
            [html.H4("💰 Total sum (€)"),
             html.P(format_number_es(s, ndigits=2))],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("📊 % insured sum"),
                html.P(format_number_es(pct_sum, ndigits=2) + "%"),
            ],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("📍 Center latitude"),
                html.P(format_number_es(lat, ndigits=6, use_comma=False)),
            ],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("📍 Center longitude"),
                html.P(format_number_es(lon, ndigits=6, use_comma=False)),
            ],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("⏱️ Total time (s)"),
                html.P(f"{format_number_es(exec_time,
                                           ndigits=6, use_comma=False)}"),
            ],
            className="kpi-card",
        ),
        html.Div(
            [
                html.H4("⚡ Algorithm time (s)"),
                html.P(f"{format_number_es(algo_time,
                                           ndigits=6, use_comma=False)}"),
            ],
            className="kpi-card",
        ),
    ]

    # 8) Build Global kpis
    idx = algo_output.get("indices", [])
    inside = df.iloc[idx]

    inside_kpis = compute_global_kpis(inside)
    inside_kpis.pop("Total Policies", None)
    global_kpi_cards = [
        *[
            html.Div(
                [html.H4(f"{EMOJIS[key]} {key}"),
                 html.P(format_number_es(value))],
                className="kpi-card",
            )
            for key, value in inside_kpis.items()
        ],
        html.Div([html.H4("—"), html.P("")], className="kpi-card"),
    ]

    # 9) Build Map: show ONLY inside points + red circle
    tile_layer = dl.TileLayer(
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution="© CartoDB",
    )

    markers = [
        dl.Marker(
            position=(row["lat"], row["lon"]),
            children=dl.Popup(f"Policy: {row['n_policy']} | "
                              f"{row['insured_sum']}€"),
        )
        for _, row in inside.iterrows()
    ]

    radius = algo_output.get("parameters", {}).get("radius", 200)
    circle = dl.Circle(center=(lat, lon), radius=radius, color="red",
                       weight=2, fill=False)

    # Configure zoom
    dlat = radius / 111320
    dlon = radius / (111320 * math.cos(math.radians(lat)))
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

    # Map
    leaflet_map = dl.Map(
        children=[tile_layer, circle, dl.LayerGroup(markers)],
        bounds=bounds,
        boundsOptions={"padding": [50, 50]},
        style={"width": "100%", "height": "100%"},
    )

    return meta, comp_table, conc_cards, global_kpi_cards, leaflet_map


# -------------------------------------------------------------------
# 10) Callback for complete CSV download.
# -------------------------------------------------------------------
@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("algo-output", "data"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def download_selected_csv(_n_clicks: int,
                          algo_output: Dict[str, Any],
                          stored_data: Dict[str, Any]) -> Any:
    """
    Download the selected subset of policies as a CSV file.

    Rebuilds the DataFrame from stored JSON, filters by the indices
    selected by the algorithm, writes the subset to disk with a
    timestamped filename, and returns it to the user via Dash.

    Parameters:
    - _n_clicks (int): Number of times the download button was clicked.
    - algo_output (Dict[str, Any]): Algorithm results containing 'indices'.
    - stored_data (Dict[str, Any]): Stored payload with 'df_json'.

    Returns:
    - Any: Dash file download response object.
    """
    # 1) Rebuild the DataFrame from the selected rows
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient="split")
    idx = algo_output.get("indices", [])
    inside = df.iloc[idx]

    # 2) Generate timestamp and output path
    ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"conc_results_{ts}.csv"
    filepath = os.path.join(reports_dir, filename)

    # 3) Save the CSV to disk
    inside.to_csv(filepath, index=False)

    # 4) Sent the CSV to the client with a generic name
    return dcc.send_data_frame(  # type: ignore[no-untyped-call,attr-defined]
        inside.to_csv, filename, index=False)


# -------------------------------------------------------------------
# 11) Callback for PDF download.
# -------------------------------------------------------------------
@callback(
    Output("download-pdf", "data"),
    Input("download-pdf-btn", "n_clicks"),
    State("stored-data", "data"),
    State("algo-output", "data"),
    prevent_initial_call=True,
)
def generate_full_pdf(_n_clicks: int,
                      stored_data: Dict[str, Any],
                      algo_output: Dict[str, Any]) -> Any:
    """Generate a comprehensive PDF report of the results.

    This function builds a multi‐page PDF including the data preview,
    global and zone KPIs, the policy map, and algorithm results.
    It fetches the stored dataset and algorithm output from memory,
    draws headers, tables, and maps, then returns the PDF bytes
    for download.

    Parameters:
    - _n_clicks (int): Click count from the download button trigger.
    - stored_data (Dict[str, Any]): Payload containing 'df_json' and
      'file_name' for the dataset.
    - algo_output (Dict[str, Any]): Results dict with keys like 'indices',
      'sum', 'lat', 'lon', 'parameters', and optional 'comparisons'.

    Returns:
    - Any: A Dash data response object containing the PDF bytes.
    """
    # --- 1) Load logo
    with open("assets/dlr_logo_black.png", "rb") as f:
        logo_bytes = f.read()
    logo_img = ImageReader(io.BytesIO(logo_bytes))

    # --- 2) Setup canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    top_margin = 0.5 * cm
    logo_w, logo_h = 4 * cm, 4 * cm

    # --- 3) First page header (logo + timestamp)
    baseline_y = draw_header(c, width, height, logo_img,
                             logo_w, logo_h, top_margin)

    # --- 4) Title on first page
    title_y = baseline_y - logo_h / 2 - 0.7 * cm
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, title_y, "SolvRisk 360 – Risk Report")

    # --- 5) Start Section 1: Data Input
    y = title_y - 1.5 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "1. Data Input")

    # 5.1) Dataset name
    y -= 0.8 * cm
    c.setFont("Helvetica", 12)
    dataset = stored_data.get("file_name", "Unknown dataset")
    c.drawString(2 * cm, y, f"Dataset used: {dataset}")

    # 5.2) Preview label
    y -= 1.0 * cm
    df = pd.read_json(io.StringIO(stored_data["df_json"]), orient="split")
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2 * cm, y, "Preview (first 3 / last 3 rows):")

    # 5.3) Table preview with monospace font
    y -= 0.6 * cm
    preview_df = pd.concat([df.head(3), df.tail(3)])
    header = preview_df.columns.tolist()
    lines = []
    rows_h = preview_df.head(3).values.tolist()
    rows_t = preview_df.tail(3).values.tolist()
    col_widths = [max(len(str(v)) for v in preview_df[col])
                  + 6 for col in header]
    fmt = ("  ").join(f"{{:<{w}}}" for w in col_widths)
    lines.append(fmt.format(*header))
    for row in rows_h:
        lines.append(fmt.format(*row))
    lines.append("…")
    for row in rows_t:
        lines.append(fmt.format(*row))

    c.setFont("Courier", 8)
    for line in lines:
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm

    y -= 0.5 * cm
    c.setLineWidth(0.5)
    c.line(2 * cm, y, width - 2 * cm, y)

    # --- 6) Start Section 2: Global KPIS
    y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "2. Global KPIs")

    global_kpis = compute_global_kpis(df)
    y -= 0.8 * cm
    c.setFont("Helvetica", 12)
    for key, value in global_kpis.items():
        formatted = format_number_es(value)
        c.drawString(2.5 * cm, y, f"• {key}: {formatted}")
        y -= 0.6 * cm

    y -= 0.4 * cm
    c.setLineWidth(0.5)
    c.line(2 * cm, y, width - 2 * cm, y)

    # --- 7) Section 3: KPIs by Zone
    y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "3. KPIs by Zone")

    y -= 1.5 * cm
    zone_df = compute_kpis_by_zone(df).copy()
    for col in zone_df.columns[1:]:
        zone_df[col] = zone_df[col].apply(lambda v: f"{v:,.2f}")

    data = [zone_df.columns.tolist()] + zone_df.values.tolist()
    n_cols = len(zone_df.columns)
    equal_w = (width - 4 * cm) / n_cols
    col_widths = [equal_w] * n_cols

    # Create and style the table
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
            ]
        )
    )

    # Draw the table
    tbl_height = len(data) * 0.45 * cm
    table.wrapOn(c, width, y)
    table.drawOn(c, 2 * cm, y - tbl_height)
    y = y - tbl_height - 1 * cm

    # --- PAGE 1: NUMBER FOOTER ---
    page = c.getPageNumber()
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 2 * cm, 1 * cm, f"Page {page}")
    c.showPage()

    # --- 8) Repeat Header (logo + timestamp) on new page
    baseline_y = draw_header(c, width, height, logo_img,
                             logo_w, logo_h, top_margin)
    y = baseline_y - logo_h / 2 - 0.7 * cm

    # --- 9) Section 4: Policy Map
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "4.Policy Map")

    y -= 0.1 * cm

    # A) Build a standalone folium map
    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=5.5,  # type: ignore[arg-type]
        tiles="CartoDB dark_matter",
        attr="© CartoDB",
    )
    from folium.plugins import MarkerCluster

    cluster = MarkerCluster().add_to(m)  # type: ignore[no-untyped-call]
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=5,
            fill=True,
            fill_color="yellow",
            fill_opacity=0.7,
            popup=f"{row['zone']}: {row['insured_sum']}",
        ).add_to(cluster)

    # B) Save to temp html & screenshot via selenium
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    m.save(tmp.name)
    file_url = "file://" + tmp.name

    png = capture_dash_map(
        url=file_url,
        css_selector="div.leaflet-container",
        width=800,
        height=600,
        wait=2,
    )

    tmp.close()
    os.unlink(tmp.name)

    # C) Draw PNG into PDF
    map_img = ImageReader(io.BytesIO(png))
    img_w = width - 4 * cm
    img_h = img_w * 0.75
    x = (width - img_w) / 2
    c.drawImage(
        map_img,
        x,
        y - img_h,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    # --- 10) Section 5: Algorithm Results
    y_map = y - img_h

    sep_y = y_map - 0.3 * cm
    c.setLineWidth(0.5)
    c.line(2 * cm, sep_y, width - 2 * cm, sep_y)

    y = sep_y - 1 * cm

    # A) Section heading
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "5. Algorithm Results")

    # B) Algorithm Metadata
    y -= 1.0 * cm
    c.setFont("Helvetica", 12)
    algo_name = algo_output.get("algorithm", "–")
    params = algo_output.get("parameters", {})

    # Algorithm name
    c.drawString(2 * cm, y, f"Algorithm used: {algo_name}")
    y -= 0.6 * cm

    # Parameters, one per line
    c.drawString(2 * cm, y, "Parameters used:")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for k, v in params.items():
        c.drawString(2.5 * cm, y, f"- {k}: {v}")
        y -= 0.5 * cm
    c.setFont("Helvetica", 12)  # restore font for what follows

    # C) Comparison table (if exists)
    comparisons = algo_output.get("comparisons", [])
    if comparisons:
        y -= 1.0 * cm
        rows = [["Algorithm", "Sum", "# policies", "Center", "Time (s)"]]
        for r in comparisons:
            center = f"({r['center'][0]:.6f}, {r['center'][1]:.6f})"
            rows.append(
                [
                    r["algo"],
                    f"{r['sum']:,.2f}",
                    str(r["count"]),
                    center,
                    f"{r['time']:.5f}",
                ]
            )
        col_width = (width - 4 * cm) / len(rows[0])
        tbl = Table(rows, colWidths=[col_width] * len(rows[0]))
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 7),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                ]
            )
        )
        tbl_h = len(rows) * 0.45 * cm
        tbl.wrapOn(c, width, y)
        tbl.drawOn(c, 2 * cm, y - tbl_h)
        y -= tbl_h + 0.5 * cm

    # --- PAGE 2 ---
    page = c.getPageNumber()
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 2 * cm, 1 * cm, f"Page {page}")
    c.showPage()

    # --- 11) New page: Results Summary (Concentration vs Global KPIs)
    baseline_y = draw_header(c, width, height, logo_img,
                             logo_w, logo_h, top_margin)
    y = baseline_y - logo_h / 2 - 0.7 * cm

    # Section heading
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, y, "Results Summary")
    y -= 1.2 * cm

    # Column X positions
    x_conc = 2 * cm
    x_glob = width / 2 + 1 * cm

    # --- 11.a) Prepare concentration data.
    conc_idx = algo_output.get("indices", [])
    conc_df = df.iloc[conc_idx]
    n_inside = len(conc_df)
    sum_inside = algo_output.get("sum", 0)
    pct_pols = n_inside / len(df) * 100
    pct_sum = sum_inside / df["insured_sum"].sum() * 100

    lat = algo_output.get("lat", 0.0)
    lon = algo_output.get("lon", 0.0)
    radius = algo_output.get("parameters", {}).get("radius", 200)
    exec_time = algo_output.get("execution_time", 0.0)
    algo_time = algo_output.get("algorithm_time", 0.0)

    # --- 11.b) Concentration Results (text)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_conc, y, "a) Concentration Results")
    y_conc = y - 0.8 * cm
    c.setFont("Helvetica", 12)
    for label, val in [
        ("Policies", f"{n_inside}"),
        ("% of total policies", f"{pct_pols:.2f}%"),
        ("Total sum (€)", f"{sum_inside:,.2f}"),
        ("% of insured sum", f"{pct_sum:.2f}%"),
        ("Center latitude", f"{lat:.6f}"),
        ("Center longitude", f"{lon:.6f}"),
        ("Total time (s)", f"{exec_time:.6f}"),
        ("Algorithm time (s)", f"{algo_time:.6f}"),
    ]:
        c.drawString(x_conc + 0.5 * cm, y_conc, f"• {label}: {val}")
        y_conc -= 0.6 * cm

    # --- 11.c) Concentration Policy Map snapshot
    # a) Create the map the same way as in display_results.
    m_conc = folium.Map(
        location=[lat, lon], zoom_start=5,
        tiles="CartoDB dark_matter", attr="© CartoDB"
    )
    # b) Add the red circle.
    folium.Circle(
        location=(lat, lon), radius=radius, color="red", weight=2, fill=False
    ).add_to(m_conc)
    # c) Add markers
    for _, row in conc_df.iterrows():
        folium.Marker(
            location=(row["lat"], row["lon"]),
            popup=f"Policy: {row['n_policy']} | {row['insured_sum']}€",
        ).add_to(m_conc)
    # d) Adjust the map to the calculated bounds.
    dlat = radius / 111320
    dlon = radius / (111320 * math.cos(math.radians(lat)))
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
    m_conc.fit_bounds(bounds, padding=(50, 50))

    # e) Capture as before.
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    m_conc.save(tmp.name)
    file_url = "file://" + tmp.name
    png_conc = capture_dash_map(
        url=file_url,
        css_selector="div.leaflet-container",
        width=800,
        height=600,
        wait=2,
    )
    tmp.close()
    os.unlink(tmp.name)

    # f) Draw the image in the PDF with the same width and aspect ratio.
    map_conc = ImageReader(io.BytesIO(png_conc))
    img_w = width - 4 * cm
    img_h = img_w * 0.75
    x_map = 2 * cm
    y_map = y_conc - img_h
    c.drawImage(
        map_conc,
        x_map,
        y_map,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        mask="auto",
    )
    y_conc = y_map - 1 * cm

    # --- 11.d) Global KPIs Results (right)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_glob, y, "b) Global KPIs Results")
    y_glob = y - 0.8 * cm
    inside_kpis = compute_global_kpis(conc_df)
    inside_kpis.pop("Total Policies", None)
    c.setFont("Helvetica", 12)
    for key, value in inside_kpis.items():
        c.drawString(x_glob + 0.5 * cm, y_glob, f"• {key}: {value:,.2f}")
        y_glob -= 0.6 * cm

    # --- PAGE 3 ---
    page = c.getPageNumber()
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 2 * cm, 1 * cm, f"Page {page}")
    c.showPage()

    # --- 12) Save results
    ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_filename = f"fire_risk_report_{ts}.pdf"
    pdf_path = os.path.join(reports_dir, pdf_filename)
    c.save()
    buf.seek(0)

    with open(pdf_path, "wb") as f:
        f.write(buf.getvalue())

    return dcc.send_bytes(  # type: ignore[no-untyped-call,attr-defined]
        buf.read(), filename=pdf_filename)
