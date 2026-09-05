-- Empirical check on the leakage rule. Code meaning "expired" must show a
-- 0% readmission rate; hospice codes are expected to be nonzero. This table
-- is the evidence cited in the methodology section.
CREATE OR REPLACE TABLE silver.disposition_audit AS
SELECT
    CAST(discharge_disposition_id AS INTEGER) AS disposition_id,
    count(*)                                  AS n_encounters,
    round(100.0 * avg(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END), 3) as pct_readmit_30d,
    CASE
        WHEN CAST(discharge_disposition_id AS INTEGER) IN (11,19,20) THEN 'excluded_expired'
        WHEN CAST(discharge_disposition_id AS INTEGER) IN (13,14)    THEN 'retained_hospice'
        ELSE 'retained'
    END AS treatment
FROM bronze.encounters
GROUP BY 1
ORDER BY pct_readmit_30d, n_encounters DESC;