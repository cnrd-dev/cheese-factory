"""
freezeML: Anomaly detection and energy prediction web app

This app gets real-time time-series data from an external system and
performs anomaly detection and energy prediction using pre-trained models.

"""

# import libraries
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# Load data from SQLite (from ETL pipeline)
engine = create_engine("sqlite:///../data/RawData.db")


# Load model (from ML pipeline)
# model = joblib.load("../models/classifier.pkl")


# Index webpage displays graphs and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # Extract data needed for graphs
    df = pd.read_sql_table("SensorData_1min", con=engine, index_col="timestamp")
    df_tmp = df[df.index > "2021-07-21"]

    # Create graphs
    graphs = [
        {
            "data": [Scatter(x=df_tmp.index, y=df_tmp.DP_temp)],
            "layout": {
                "title": "Temperature for the Dispatch Cold Room",
                "yaxis": {"title": "Temperature"},
                "xaxis": {"title": "Time"},
                # "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
            },
        },
        {
            "data": [Scatter(x=df_tmp.index, y=df_tmp.WIP_temp)],
            "layout": {
                "title": "Temperature for the WIP Cold Room",
                "yaxis": {"title": "Temperature"},
                "xaxis": {"title": "Time"},
                # "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
            },
        },
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template("index.html", ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route("/energy")
def go():
    # Extract data needed for graphs
    df = pd.read_sql_table("SensorData_1hour", con=engine, index_col="timestamp")
    df_tmp = df[df.index > "2021-07-01"]

    # Create graphs
    graphs = [
        {
            "data": [Scatter(x=df_tmp.index, y=df_tmp.DP_energy)],
            "layout": {
                "title": "Energy consumption for the Dispatch Cold Room",
                "yaxis": {"title": "Energy (kWh)"},
                "xaxis": {"title": "Time"},
                # "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
            },
        },
        {
            "data": [Scatter(x=df_tmp.index, y=df_tmp.WIP_energy)],
            "layout": {
                "title": "Energy consumption for the WIP Cold Room",
                "yaxis": {"title": "Energy (kWh)"},
                "xaxis": {"title": "Time"},
                # "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
            },
        },
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template("energy.html", ids=ids, graphJSON=graphJSON)


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
