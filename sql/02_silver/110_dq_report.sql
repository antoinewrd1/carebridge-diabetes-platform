CREATE OR REPLACE TABLE silver.dq_report AS
SELECT 'bronze_encounter_rows' as metric, count(*) AS value FROM bronze.encounters
UNION ALL SELECT 'silver_encounter_rows', count(*) FROM silver.encounters
UNION ALL SELECT 'dropped_expired',
    (SELECT count(*) FROM bronze.encounters
    WHERE CAST(discharge_disposition_id AS INTEGER) IN (11,19,20))
UNION ALL SELECT 'retained_hospice', sum(CASE WHEN is_hospice THEN 1 ELSE 0 END)
    FROM silver.encounters
UNION ALL SELECT 'dropped_unknown_gender',
    (SELECT count(*) FROM bronze.encounters WHERE gender NOT IN ('Male', 'Female'))
UNION ALL SELECT 'distinct_patients', count(DISTINCT patient_nbr) FROM silver.encounters
UNION ALL SELECT 'readmit_30d_positives', sum(readmitted_30d) FROM silver.encounters
UNION ALL SELECT 'null_race', count(*) FROM silver.encounters WHERE race IS NULL
UNION ALL SELECT 'null_payer_code', count(*) FROM silver.encounters WHERE payer_code IS NULL
UNION ALL SELECT 'null_medical_specialty', count(*) FROM silver.encounters WHERE medical_specialty IS NULL
UNION ALL SELECT 'a1c_tested', sum(a1c_tested) FROM silver.encounters
UNION ALL SELECT 'brfss_rows', count(*) FROM silver.brfss;