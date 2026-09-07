CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.fct_encounter AS
WITH icd AS (
    SELECT
        *,
        -- ICD-9 chapter grouping. V and E codes are supplementary
        -- classifications, not numeric diagnoses, so they branch first.
        CASE
            WHEN diag_1 IS NULL THEN 'unknown'
            WHEN diag_1 LIKE 'V%' OR diag_1 LIKE 'E%' THEN 'supplementary'
            WHEN FLOOR(TRY_CAST(diag_1 AS DOUBLE)) = 250 THEN 'diabetes'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 390 AND 459 THEN 'circulatory'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 460 AND 519 THEN 'respiratory'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 520 AND 579 THEN 'digestive'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 580 AND 629 THEN 'genitourinary'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 710 AND 739 THEN 'musculoskeletal'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 800 AND 999 THEN 'injury'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 140 AND 239 THEN 'neoplasms'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 001 AND 139 THEN 'infectious'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 240 AND 279 THEN 'endocrine_other'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 290 AND 319 THEN 'mental'
            WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 320 AND 389 THEN 'nervous_sensory'
            ELSE 'other'
        END AS diag_1_group
    FROM silver.encounters
),
seq AS (
    SELECT
        *,
        row_number() OVER (PARTITION BY patient_nbr ORDER BY encounter_id) AS encounter_seq,
        count(*)     OVER (PARTITION BY patient_nbr)                       AS patient_encounter_count
    FROM icd
)
SELECT
    encounter_id,
    patient_nbr,
    race, gender, age_band,
    CAST(regexp_extract(age_band, '\[(\d+)-', 1) AS INTEGER) + 5 AS age_midpoint,

    time_in_hospital, num_lab_procedures, num_procedures, num_medications,
    number_outpatient, number_emergency, number_inpatient, number_diagnoses,
    number_outpatient + number_emergency + number_inpatient AS prior_visits_total,

    admission_type_id, discharge_disposition_id, admission_source_id,
    medical_specialty, payer_code,
    diag_1, diag_1_group,
    max_glu_serum, a1c_result, a1c_tested, med_changed, on_diabetes_med,
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride,
    acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone,
    rosiglitazone, acarbose, miglitol, troglitazone, tolazamide,
    insulin,
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
    is_hospice,

    encounter_seq, patient_encounter_count,
    readmitted, readmitted_30d
FROM seq;