# Outputs

Everything the pipeline produced for the paper, so the reported numbers can be recomputed without API calls
(`python analysis/paper_numbers.py`). The pipeline writes here (`pipeline/config.py`: `WORK_DIR`); relative paths given
on the command line (e.g. `--out queries/et_expand.csv`) are inside this folder.

| Folder | Contents |
|---|---|
| `splits/folds.csv` | fold of every code: fold 1 = development (127 codes), folds 2–5 = test (511 codes) |
| `queries/` | search inputs per route: `et.csv`, `en.csv` (names), `et_translate.csv`, `*_expand*.csv` (LLM phrases; `_r2`, `_r3` = repeat runs; `_gpt4o`, `_gpt4omini` = other LLMs). Next to each LLM route: `.jsonl` raw LLM responses and `.json` run metadata (model snapshot, settings, token use). `dev/` = prompt development on fold 1 |
| `runs/retrieval/` | `<route>__<embedding>-<dimensions>[__mean].csv`: ranked candidates per code (`source_code, rank, concept_id, score`), **trimmed to the top 50** (the analysis uses at most 50; `retrieve.py` writes 300) |
| `runs/rerank/` | `<route>__<embedding>__k<k>__<effort or llm>__<prompt>_f<folds>.csv`: the LLM's selected concept per code, with `.jsonl` raw responses and `.json` metadata. Paper runs are `rerank.v2` on folds 2345 (`_f2345`) |
| `runs/zero_shot/` | LLM without retrieval (not reported in the final paper) |
| `runs/hierarchy/` | distance of each prediction to the nearest reference concept (`hierarchy_report.py`) and `summary.csv` |
| `usagi/` | Usagi inputs (`usagi_input_*.csv`), Usagi's automatic top suggestions (`usagi_output_*.csv`; see the main README) and the Google Translate output used for one input (`google/`) |

Not included (ignored by git, regenerable): `index/` (embeddings of the SNOMED CT concept names, `build_index.py`),
`queries/cache/` (query embeddings) and `logs/`.

Routes: `et` / `en` = unmodified Estonian / official English names; `et_translate` = LLM translation;
`et_expand` / `en_expand` = contextual expansion. Concept ids are OMOP concept ids (see `data/vocabulary/`).
