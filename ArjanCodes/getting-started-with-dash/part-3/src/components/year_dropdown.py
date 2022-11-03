import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from . import ids
import i18n
from ..data.source import DataSource

def render(app: Dash, source: DataSource) -> html.Div:
    # all_years: list[str] = data[DataSchema.YEAR].tolist()
    # unique_years = sorted(set(all_years), key=int)

    @app.callback(
        Output(ids.YEAR_DROPDOWN, "value"),
        Input(ids.SELECT_ALL_YEARS_BUTTON, "n_clicks"),
    )
    def select_all_years(_: int) -> list[str]:
        return source.unique_years

    return html.Div(
        children=[
            html.H6(i18n.t("general.year")),
            dcc.Dropdown(
                id=ids.YEAR_DROPDOWN,
                options=[{"label": year, "value": year} for year in source.unique_years],
                value=source.unique_years,
                multi=True,
            ),
            html.Button(
                className="dropdown-button",
                children=[i18n.t("general.select_all")],
                id=ids.SELECT_ALL_YEARS_BUTTON,
                n_clicks=0,
            ),
        ]
    )
