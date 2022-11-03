import pandas as pd
import i18n
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.components import ids
from ..data.loader import DataSchema
from ..data.source import DataSource


def render(app: Dash, source: DataSource) -> html.Div:
    @app.callback(
        Output(ids.MONTH_DROPDOWN, "value"),
        [
            Input(ids.YEAR_DROPDOWN, "value"),
            Input(ids.SELECT_ALL_MONTHS_BUTTON, "n_clicks"),
        ],
    )  
    def select_all_months(years: list[str], _: int) -> list[str]:
        return source.filter(years=years).unique_months
    
    return html.Div(
        children=[
            html.H6(i18n.t("general.month")),
            dcc.Dropdown(
                id = ids.MONTH_DROPDOWN,
                options=[{"label": month, "value": month} for month in source.unique_months],
                value=source.unique_months,
                multi=True,
            ),
            html.Button(className="dropdown-button",
                        children=[i18n.t("general.select")],
                        id=ids.SELECT_ALL_MONTHS_BUTTON,
                        n_clicks=0,
            ),
        ]
    )
