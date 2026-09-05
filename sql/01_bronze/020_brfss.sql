CREATE OR REPLACE TABLE bronze.brfss AS
SELECT
    *,
    current_timestamp AS ingested_at,
    'kaggle:alexteboul/diabetes-health-indicators-dataset' AS _source
FROM read_csv_auto(
    '{{raw_dir}}/brfss_indicators/diabetes_binary_health_indicators_BRFSS2015.csv',
    header = true
);