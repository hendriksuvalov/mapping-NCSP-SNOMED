# Cross-lingual mapping of surgical procedure codes to SNOMED CT: an Estonian–English evaluation

Code, reference mappings and model outputs for the MIE 2027 paper by Oskar Telgmaa, Sirli Tamm and Hendrik Šuvalov
(Institute of Computer Science, University of Tartu).

We map codes of the NOMESCO Classification of Surgical Procedures (NCSP) as used in Estonia to standard SNOMED CT
concepts and compare Estonian with official English procedure names for the same codes:

1. **Search input**: the unmodified Estonian or English name, an LLM translation of the Estonian name, or a
   contextual expansion (six or seven English SNOMED CT-style search phrases). Translation and expansion use the
   code's NCSP hierarchy as context.
2. **Retrieval**: embedding search over the 71,181 standard SNOMED CT concepts of class Procedure; the 30 most
   similar concept names are kept (for expansions, mean cosine similarity over the phrases).
3. **Selection**: the LLM picks one candidate, seeing the original name and its NCSP hierarchy.

Baselines are Usagi's automatic top suggestion and retrieval alone. Evaluation is on 511 held-out codes (folds 2–5);
prompts were developed on 127 codes (fold 1). The evaluation design is documented in [PROTOCOL.md](PROTOCOL.md).

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | the pipeline, baselines and evaluation (`evaluate.py`, `hierarchy.py`), with unit tests |
| `analysis/` | `paper_numbers.py` prints every number in the paper; `results.py` computes them; `export_figures.py` draws Figure 1 |
| `data/reference/` | the reference mappings (chapters L, M, T) and the NCSP names and hierarchy ([README](data/reference/README.md)) |
| `data/vocabulary/` | place for the OMOP vocabulary, **not included** ([README](data/vocabulary/README.md)) |
| `outputs/` | folds, LLM queries and raw responses, retrieval candidates, selections, hierarchy distances, Usagi runs ([README](outputs/README.md)) |

## Setup

```
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the OMOP vocabulary (with SNOMED) from [Athena](https://athena.ohdsi.org) into `data/vocabulary/`; the
paper used release v5.0 29-AUG-26. Everything else needed for the analysis is in the repository.

## Reproducing the paper's numbers (no API calls)

```
python -m pytest pipeline                # unit tests of the evaluation code
python analysis/paper_numbers.py         # Table 1, all differences with 95% CIs, Figure 1 values
python analysis/export_figures.py        # Figure 1 -> analysis/figures/hierarchy.pdf
```

`paper_numbers.py` reads the model outputs in `outputs/` and needs only the vocabulary.

## Re-running the pipeline (Azure OpenAI)

Models: `text-embedding-3-large` and `-small`, GPT-6 Luna (`gpt-6-luna`, 2026-09-22), GPT-4o (2024-11-20) and
GPT-4o mini (2024-07-18) as Azure OpenAI deployments; names are in `pipeline/config.py`. Put the API key in
`.azure_key` at the repository root or set `AZURE_OPENAI_API_KEY`; set `AZURE_OPENAI_ENDPOINT` for your endpoint.
Model outputs vary between runs, so re-running reproduces the procedure rather than the exact numbers.

Run from `pipeline/`; relative paths on the command line refer to `outputs/`. Every LLM script logs responses as they
arrive and resumes after an interruption.

```
python make_folds.py                                     # splits/folds.csv
python make_queries.py                                   # queries/en.csv, queries/et.csv
python build_index.py --model large --out index/large    # embeds the SNOMED CT procedure concepts (also: small)
python llm_queries.py --prompt et_translate.v2 --out queries/et_translate.csv
python llm_queries.py --prompt et_expand.v2 --out queries/et_expand.csv      # also en_expand.v2
python retrieve.py --route et en et_translate --variants large-3072 small-1536
python retrieve.py --route et_expand en_expand --variants large-3072 small-1536 --pooling mean
python rerank.py --route et_expand --variant large-3072 --k 30 --prompt rerank.v2 --reasoning-effort low --folds 2 3 4 5
python hierarchy_report.py                               # distances to the reference in the SNOMED CT hierarchy
```

`run_plan.py` lists the exact runs behind the paper (repeats, other models, numbers of candidates). `map_codes.py`
applies the pipeline (expanded official English name, large embeddings, GPT-6 Luna) to new NCSP codes listed in Excel
files.

## Usagi baseline

We used [Usagi](https://ohdsi.github.io/Usagi/) v1.4.3 ([installation](https://ohdsi.github.io/Usagi/installation.html))
with its index built from the same vocabulary release. The inputs are in `outputs/usagi/` (`usagi_input_*.csv`: Google-
translated Estonian names, LLM-translated names, official English names; made by `make_usagi_inputs.py` and
`make_usagi_google_input.py`). They were imported with `source_code`, `source_name` and `source_frequency` as the code,
name and frequency columns, with the filters *standard concepts*, vocabulary *SNOMED* and concept class *Procedure*,
and all other settings at their defaults. Usagi's automatic top suggestions, without any review, were saved as
`usagi_output_*.csv` and are scored with `python pipeline/score_usagi.py outputs/usagi/usagi_output_*.csv`.

## Notes

- SNOMED CT concept names in the reference mappings and outputs are part of SNOMED CT (© SNOMED International).
  The OMOP vocabulary is distributed by Athena under its own terms and is not included.
- Citation: to be added after publication. Licence: to be decided by the authors.
