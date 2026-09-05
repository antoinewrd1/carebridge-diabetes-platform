CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE TABLE bronze.encounters AS
SELECT
    *,
    current_timestamp AS ingested_at,
    'kaggle:brandao/diabetes' AS _source
FROM read_csv_auto(
    '{{raw_dir}}/diabetes_encounters/diabetic_data.csv',
    header = true,
    all_varchar = true
);