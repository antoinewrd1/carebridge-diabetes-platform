CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE TABLE silver.encounters AS
WITH typed AS (
    SELECT
        CAST(encounter_id AS BIGINT)                          AS encounter_id,
        CAST(patient_nbr  AS BIGINT)                          AS patient_nbr,

        NULLIF(race, '?')                                     AS race,
        CASE WHEN gender IN ('Male','Female') THEN gender END AS gender,
        age                                                   AS age_band,

        CAST(admission_type_id        AS INTEGER)             AS admission_type_id,
        CAST(discharge_disposition_id AS INTEGER)             AS discharge_disposition_id,
        CAST(admission_source_id      AS INTEGER)             AS admission_source_id,
        CAST(time_in_hospital         AS INTEGER)             AS time_in_hospital,

        NULLIF(medical_specialty, '?')                        AS medical_specialty,
        NULLIF(payer_code, '?')                               AS payer_code,

        CAST(num_lab_procedures AS INTEGER)                   AS num_lab_procedures,
        CAST(num_procedures     AS INTEGER)                   AS num_procedures,
        CAST(num_medications    AS INTEGER)                   AS num_medications,
        CAST(number_outpatient  AS INTEGER)                   AS number_outpatient,
        CAST(number_emergency   AS INTEGER)                   AS number_emergency,
        CAST(number_inpatient   AS INTEGER)                   AS number_inpatient,
        CAST(number_diagnoses   AS INTEGER)                   AS number_diagnoses,

        NULLIF(diag_1, '?')                                   AS diag_1,
        NULLIF(diag_2, '?')                                   AS diag_2,
        NULLIF(diag_3, '?')                                   AS diag_3,

        NULLIF(max_glu_serum, 'None')                         AS max_glu_serum,
        NULLIF("A1Cresult", 'None')                           AS a1c_result,

        insulin, metformin, glipizide, glyburide,
        pioglitazone, rosiglitazone,
        "change"                                              AS med_change,
        "diabetesMed"                                         AS on_diabetes_med,
        readmitted
    FROM bronze.encounters
),
flagged AS (
    SELECT
        *,
        -- LEAKAGE GUARD. Codes 11, 19, 20 mean the patient died: readmission
        -- is structurally impossible. Verified empirically -- all three show a
        -- 0.000% observed 30-day readmission rate across 1,652 encounters.
        discharge_disposition_id IN (11, 19, 20)             AS is_expired,

        -- Codes 13, 14 mean hospice. These patients are ALIVE and are
        -- readmitted at 4.762% and 6.452% against an 11.16% baseline.
        -- Retained as a covariate rather than excluded.
        discharge_disposition_id IN (13, 14)                 AS is_hospice,

        CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END       AS readmitted_30d,
        CASE WHEN a1c_result IS NOT NULL THEN 1 ELSE 0 END   AS a1c_tested,
        CASE WHEN med_change = 'Ch'  THEN 1 ELSE 0 END       AS med_changed
    FROM typed
)
SELECT * FROM flagged
WHERE NOT is_expired
  AND gender IS NOT NULL;