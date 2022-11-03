from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
import i18n
from src.components.layout import create_layout
from src.data.loader import load_transaction_data
from src.data.source import DataSource

DATA_PATH = "./data/transactions.csv"
LOCALE = "en"

def main() -> None:
    """_summary_
    """
    i18n.set("locale", LOCALE)
    # Check to make sure the library is loading the files
    i18n.load_path.append("locale")
    data = load_transaction_data(DATA_PATH, LOCALE)
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = i18n.t("general.app_title")
    app.layout = create_layout(app, DataSource(data))
    app.run()
    
if __name__ == "__main__":
    main()