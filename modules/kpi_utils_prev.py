"""KPI computations and 3D scatter plot generation for insured capital."""
import plotly.graph_objects as go
from typing import Dict, Union
import pandas as pd


def compute_global_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute global descriptive statistics for the 'insured_sum' column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least the
     'insured_sum' column.

    Returns:
    - dict: Dictionary containing the computed KPIs.
    """
    insured = df["insured_sum"]

    kpis = {
        "Total Policies": len(insured),
        "Minimum (€)": insured.min(),
        "1st Quartile (€)": insured.quantile(0.25),
        "Median (€)": insured.median(),
        "3rd Quartile (€)": insured.quantile(0.75),
        "Mean (€)": insured.mean(),
        "Maximum (€)": insured.max(),
        "St.Desv. (€)": insured.std(),
    }

    return kpis


def compute_kpis_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KPIs grouped by the 'zone' column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'insured_sum' and 'zone' columns.

    Returns:
    - pd.DataFrame: A DataFrame of KPIs per zone.
    """
    grouped = df.groupby("zone")["insured_sum"].agg(
        [
            ("Total Policies", "count"),
            ("Minimum (€)", "min"),
            ("1st Quartile (€)", lambda x: x.quantile(0.25)),
            ("Median (€)", "median"),
            ("3rd Quartile (€)", lambda x: x.quantile(0.75)),
            ("Mean (€)", "mean"),
            ("Maximum (€)", "max"),
            ("St.Desv. (€)", "std"),
        ]
    )

    return grouped.reset_index()


def generate_3d_scatter(df: pd.DataFrame,
                        mode: str = "global"
                        ) -> Union[go.Figure, Dict[str, go.Figure]]:
    """3D scatter plot function.

    Generate a 3D scatter plot of latitude, longitude, and insured capital.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'lat', 'lon', 'insured_sum',
         and 'zone' columns.
        mode (str): "global" for a single plot of all zones, "per_zone"
        to return one plot per zone.

    Returns:
        plotly.graph_objs.Figure or dict[str, Figure]: One 3D plot or
        a dictionary of plots by zone.
    """
    if mode == "global":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=df["lat"],
                y=df["lon"],
                z=df["insured_sum"],
                mode="markers",
                marker=dict(
                    size=3, color=df["insured_sum"],
                    colorscale="Plasma", opacity=0.8
                ),
                text=df.get("zone", None),
                name="All Zones",
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Latitude",
                yaxis_title="Longitude",
                zaxis_title="Capital (€)",
            ),
            title="Global 3D Distribution of Insured Capital",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

    elif mode == "per_zone":
        zone_figs = {}
        for zone in df["zone"].unique():
            sub_df = df[df["zone"] == zone]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=sub_df["lat"],
                    y=sub_df["lon"],
                    z=sub_df["insured_sum"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=sub_df["insured_sum"],
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    text=sub_df["zone"],
                    name=zone,
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Latitude",
                    yaxis_title="Longitude",
                    zaxis_title="Capital (€)",
                ),
                title=f"3D Distribution in {zone.capitalize()}",
                margin=dict(l=0, r=0, b=0, t=40),
            )
            zone_figs[zone] = fig
        return zone_figs

    else:
        raise ValueError("Mode must be either 'global' or 'per_zone'")
