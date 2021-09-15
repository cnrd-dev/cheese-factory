# Making better cheese with IoT and ML

A small cheese factory has recently implemented an IoT solution to provide real-time monitoring of their refrigeration system. The system measures temperature inside the cold room, the ambient temperature and the power consumption of the refrigeration unit. Instead of manually taking spot measurement every few hours for quality control, the real-time temperature is now captured automatically every 30 to 60 seconds which provides more consistent monitoring. Alarms are sent out if the temperature exceeds the upper or lower limits, however, the refrigeration system includes an automatic defrost cycle which causes some nuisance alarms.

## Motivation for optimisation

Reduce nuisance alarms when monitoring refrigeration temperature and use the power consumption data to determine energy saving opportunities.

Three key business questions:

1. How can we better implement refrigeration temperature monitoring to avoid nuisance alarms due to defrosting cycles?
1. What is the current energy performance of the refrigeration system?
1. What are the potential energy savings?

This Data Science project is using anomaly detection for refrigeration temperature monitoring and regression models to determine energy performance of the refrigeration system. I have written a [blog post](https://medium.com/@coenraad-pretorius/5e9cccc63c3f) on Medium which gives an overview of the project and discusses the results.

## Optimisation process

- Define the business problems to be solved
- Build data ingestion pipeline for analysis and training models
- Perform EDA & ML to build the appropriate models
- Develop and API for deployment to integrate with the real-time monitoring system (phase 2)

## File descriptions

- Data ETL notebook: `data-etl.ipynb`
- EDA: `eda.ipynb`
- Anomaly notebooks: `anomaly-detection-wip.ipynb` and `anomaly-detection-dp.ipynb`
- Energy modelling: `energy-model.ipynb`
- cnrdlib: Custom package containing common functions used for data engineering and analysis.

## How to run the notebooks

Dependencies and virtual environment details are located in the `Pipfile` which can be used with `pipenv`.

## License

GNU GPL v3

## Author

Coenraad Pretorius

## Acknowledgements

#### Timeseries anomaly detection using an Autoencoder

**Author:** [pavithrasv](https://github.com/pavithrasv)<br>
**Article:** [https://keras.io/examples/timeseries/timeseries_anomaly_detection/](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)
