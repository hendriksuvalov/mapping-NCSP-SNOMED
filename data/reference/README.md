# Reference data

| File | Contents |
|---|---|
| `NCSP_AUG26/source_to_concept_map_NCSP_L_M_chapters.csv` | reference mappings of NCSP chapters L (female genital organs) and M (obstetric procedures) |
| `NCSP_AUG26/source_to_concept_map_NCSP_T_chapter.csv` | reference mappings of NCSP chapter T (minor surgical procedures) |
| `NCSP/ncsp_classifier.csv` | Estonian NCSP: code, Estonian name (`name`), official English name (`english_name`) and hierarchy (`parent0`–`parent2`) |

**Reference mappings.** OMOP `SOURCE_TO_CONCEPT_MAP`-style tables, one row per (NCSP code, target concept). Columns used by
the pipeline: `source_code`, `source_code_description` (Estonian name), `target_concept_id` (OMOP concept id of the
SNOMED CT target), `est_target_concept_name`. A code with several rows has several targets. `target_concept_id = 0`
means the code is not mapped; such codes are excluded from the evaluation (770 codes in the files, 638 mapped:
L 294, M 111, T 233). The mappings were created by a health informatics specialist, reviewed by a medical expert and
fixed (release of 18 September 2026) before the experiments.

**NCSP.** The NOMESCO Classification of Surgical Procedures as used in Estonia, from the Publishing Centre of the
Health and Welfare Information Systems Centre (TEHIK), https://pub.e-tervis.ee/classifications/NCSP.

SNOMED CT concept names in these files are part of SNOMED CT (© SNOMED International); concept ids refer to the OMOP
vocabulary (see `../vocabulary/README.md`).
