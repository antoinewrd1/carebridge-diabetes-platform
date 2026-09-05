-- Repeat patients violate the independence assumption: 100,111 encounters
-- come from 70,346 patients. The primary analysis keeps the FIRST encounter
-- per patient. A sensitivity analysis uses all encounters with
-- patient-grouped cross-validation.
CREATE OR REPLACE TABLE gold.ml_cohort AS
SELECT * FROM gold.fct_encounter WHERE encounter_seq = 1;