-- BRFSS arrives clean: all numeric, no sentinels. Silver's job here is
-- naming, explicit typing, and asserting the file variant is correct.
CREATE OR REPLACE TABLE silver.brfss AS
SELECT
    CAST("Diabetes_binary"      AS TINYINT)  AS has_diabetes,
    CAST("Stroke"               AS TINYINT)  AS has_stroke,
    CAST("HeartDiseaseorAttack" AS TINYINT)  AS has_cardiac,

    CAST("BMI"        AS DOUBLE)   AS bmi,
    CAST("HighBP"     AS TINYINT)  AS high_bp,
    CAST("HighChol"   AS TINYINT)  AS high_chol,
    CAST("CholCheck"  AS TINYINT)  AS chol_checked,
    CAST("Smoker"     AS TINYINT)  AS smoker,
    CAST("PhysActivity" AS TINYINT) AS phys_activity,
    CAST("Fruits"     AS TINYINT)  AS eats_fruit,
    CAST("Veggies"    AS TINYINT)  AS eats_veg,
    CAST("HvyAlcoholConsump" AS TINYINT) AS heavy_alcohol,

    CAST("AnyHealthcare" AS TINYINT) AS has_coverage,
    CAST("NoDocbcCost"   AS TINYINT) AS skipped_care_cost,

    CAST("GenHlth"   AS TINYINT) AS gen_health,
    CAST("MentHlth"  AS TINYINT) AS mental_health_days,
    CAST("PhysHlth"  AS TINYINT) AS phys_health_days,
    CAST("DiffWalk"  AS TINYINT) AS difficulty_walking,

    CAST("Sex"       AS TINYINT) AS sex,
    CAST("Age"       AS TINYINT) AS age_band,
    CAST("Education" AS TINYINT) AS education,
    CAST("Income"    AS TINYINT) AS income_band
FROM bronze.brfss;

