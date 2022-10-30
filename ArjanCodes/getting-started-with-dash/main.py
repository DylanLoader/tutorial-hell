from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout

def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Medal Dashboard"
    app.layout = create_layout(app) # Set placeholder layout
    app.run()
    
if __name__ == "__main__":
    main()