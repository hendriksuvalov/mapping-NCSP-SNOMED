# OMOP vocabulary (not included)

The pipeline and the analysis need the OMOP standardised vocabularies, which are licensed and cannot be redistributed.

1. Create a free account at https://athena.ohdsi.org and download a vocabulary bundle that includes **SNOMED**.
   The paper used release **v5.0 29-AUG-26** (shown in `VOCABULARY.csv`); other releases work but may change results.
2. Unzip it here, so that this folder contains `CONCEPT.csv`, `CONCEPT_ANCESTOR.csv`, `CONCEPT_RELATIONSHIP.csv`,
   `CONCEPT_SYNONYM.csv`, `VOCABULARY.csv`, … (tab-separated, as delivered by Athena).

Everything in this folder except this README is ignored by git.
