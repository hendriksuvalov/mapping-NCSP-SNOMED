# Evaluation protocol — NCSP → SNOMED CT mapping

Working files named below (`splits/`, `index/`, `queries/`, `runs/`) live in `outputs/`.

Agreed 2026-09-25, revised 2026-09-27 (see changelog). Changes to this file after results exist must be noted in
the changelog below.

## Data
- **Gold:** `data/reference/NCSP_AUG26/` (source repository commit `d140d49`), chapters L, M, T.
  Codes with `target_concept_id = 0` are unmapped and excluded.
- **Vocabulary:** Athena `v5.0 29-AUG-26` in `data/vocabulary/`.
- **Index:** standard SNOMED concepts with `concept_class_id = Procedure`, any OMOP domain (71,181 concepts),
  embedded by concept name with `text-embedding-3-small` (1536 d) and `text-embedding-3-large` (3072 d); built
  2026-09-27, manifests in `index/*/manifest.json`.
- **Dimension variants** (e.g. 1024, 1536, 2048 for large) are obtained by truncating the full vectors and
  re-normalising, not by separate API requests. Check (1,200 concepts, 254 fold-1 queries): identical top-1
  retrieval to the API's `dimensions` parameter, top-10 overlap ≥ 99.8%, differences within repeat-request noise.
  Tables show only the dimension size; the methods section states the truncation in one sentence with a reference
  to OpenAI's embedding documentation (optionally Kusupati et al., 2022, Matryoshka Representation Learning).
- After the index filter: **638 codes** = 577 1:1 + 55 1:n + 6 unanswerable
  (MAB20, MAF20, TNG31, TNG40, TNX39, TBA20: no gold target is a procedure concept).

## Development and test split
- `splits/folds.csv` from `make_folds.py` (seed `20260925`): 5 folds of 127–128 codes. Codes sharing the first
  three NCSP characters stay together (37% of codes share a gold target with such a sibling); folds are balanced by
  chapter × mapping type.
- **Development set = fold 1** (127 codes: 115 1:1, 11 1:n, 1 unanswerable). All prompt development and any choice
  made from results happen here. Development results are reported only in the appendix.
- **Test set = folds 2–5** (511 codes: 462 1:1, 44 1:n, 5 unanswerable). Never used for any decision. All results
  in the article come from the test set.
- Why not cross-validation (the earlier plan): prompts were developed by hand on fold 1, so fold 1 cannot serve as a
  test fold, and redeveloping prompts five times by hand is not realistic. The rule itself is unchanged: no code we
  report a result for may have influenced any decision.

## Query routes
- `en`: official English NCSP name; `et`: original Estonian name.
- `et_translate`: LLM translation of the Estonian name into English.
- `et_expand`, `en_expand`: the LLM produces several English SNOMED-style phrasings (preferred term, synonyms, a more
  general name, a broad name, a one-sentence description); each is embedded and a concept's score is its **mean**
  similarity over the phrasings.
- **Frozen (2026-09-27) after two iterations on fold 1:** `et_translate.v2`, `et_expand.v2`, `en_expand.v2`
  (prompts in `llm_queries.py`; they include the NCSP hierarchy path as context). Max pooling let broad phrasings
  take rank 1; mean and centroid give identical rankings. v1 prompts and max pooling are development history only.
- Retrieval saves the top 300 candidates per code (`runs/retrieval/`); the embedding-only baseline is the top-1 candidate.

## Reranking
- The LLM sees the original NCSP name in the route's source language plus the NCSP hierarchy path (design B), and
  the top-k candidates as a numbered list; it answers with a list number. Invalid answers are retried (3 attempts);
  a code still without a valid answer is a miss and is flagged (no fallback to the first candidate).
- **Frozen prompt `rerank.v2`** (in `rerank.py`) after two iterations on fold 1: v2 adds an explicit "no added
  detail" rule. Fold 1, k=30: et_expand 59.8 → 66.1%, en_expand 62.2 → 62.2%. Two identical v2 runs differed by
  0–2 codes (87–94% identical choices).

## Fixed settings (chosen in advance)
- **Main configuration:** `text-embedding-3-large` (3072 d) + GPT-6 Luna (`gpt-6-luna`) for every LLM step,
  reasoning effort `low`, temperature at the provider default (Luna cannot change it), k = 30 candidates.
- **Model comparison (RQ1):** Estonian expanded input in all four combinations of {large, small} embeddings ×
  {Luna, GPT-4o}. With GPT-4o both the expansion and the choice are done by GPT-4o, with the same frozen prompts,
  provider-default temperature and no reasoning-effort setting (not supported). More LLMs may be added later.
- **Appendix checks:** k = 10 and 50 (Estonian expanded); a variant returning up to 2 concepts (prompt developed on
  fold 1); embedding size (search only).

## Task and metrics
- The main pipeline returns **one** concept per NCSP code.
- Gold set *G* = the code's targets that are in the index; prediction set *P*.
- **Liberal accuracy:** correct if *P ∩ G ≠ ∅*.
- **Conservative:** precision = |*P ∩ G*| / |*P*|, recall = |*P ∩ G*| / |*G*|, F1; macro-averaged over codes.
  "All answers found" = *G ⊆ P*. With one prediction, precision equals liberal accuracy.
- **Denominator:** all codes of the evaluated set (511 for test). Unanswerable codes stay in and count as misses.
- Report overall, per chapter (L/M/T) and per mapping type (1:1 / 1:n); recall@k for retrieval.
- **Hierarchy distance** (`hierarchy.py`, `hierarchy_report.py`): steps from the chosen concept and from the closest
  correct concept up to their nearest common ancestor in OMOP `CONCEPT_ANCESTOR`, added together. Reported as R0
  (exact), R1, R2, R3 and "further" (4+ steps, unrelated, or not in the hierarchy), cumulative "within R1/R2", and for
  near misses (R1-R2) whether the choice was more general, more specific or lateral. Cut-off (R3 shown, 4+ = further)
  set on fold 1 by inspecting examples at each distance; no weighted partial-credit score.

## Rules
1. **Prompt development and every choice made from results** use the development set (fold 1) only.
2. **Configurations are declared here before running**; additions after seeing results go in the changelog.
3. **Few-shot examples or known mappings** used by the pipeline may only come from the development set.
4. **Baselines** (Usagi, LLM without search, embedding-only top-1) are scored with the same evaluator on the same sets.
5. **Uncertainty:** 95% CIs by cluster bootstrap over NCSP groups (first three characters); paired comparisons
   with McNemar's test on the same codes.
6. **Repeats:** the two main configurations (Estonian and English expanded, main models) are run 3 times end to
   end, including new expansions; report mean and range. Other rows are single runs and labelled as such.
7. **Logging:** every run writes a manifest (config, deployment and returned model snapshot, API version,
   timestamps, token counts, failed-call counts).
8. **API quota:** the API had a call quota; scripts stop on a quota error and
   resume failed codes on rerun.

## Assumptions about the gold data
- Gold mappings are taken as-is, including questionable ones (e.g. TBA20, TNG40): they are the best available annotations.
- For codes with several targets, the main pipeline still returns one concept; results are reported both ways
  (any hit counts / all targets required).
- T-chapter mappings are assumed to be expert annotations made without tool suggestions.

## Changelog
- 2026-09-25: initial version (5-fold cross-validation with a configuration sweep).
- 2026-09-27: added gold-data assumptions (decided in absence of annotator feedback).
- 2026-09-27: indexes built; dimension variants by truncation (agreed).
- 2026-09-27: added query routes (translation and semantic expansion); Usagi postponed.
- 2026-09-27: froze v2 translation/expansion prompts and mean pooling (fold-1 development).
- 2026-09-27: reranker design B and prompt rerank.v2 frozen.
- 2026-09-27: **replaced cross-validation with a plain split** (fold 1 development, folds 2–5 test) because prompts
  were developed by hand on fold 1; dropped the configuration sweep and leave-one-chapter-out; fixed k = 30 and
  reasoning effort `low` in advance; added the model comparison (large/small × Luna/GPT-4o). No test-set results
  of the reranker existed at this point.
- 2026-09-28: day 1 run on the test set (decision (a): keep rerank.v2 as frozen); added the hierarchy distance
  metric (R0-R3, direction of near misses), cut-off set on fold 1.
- 2026-09-30: appendix variant `rerank_multi.v1` (may return two concepts; example taken from development codes
  only) frozen after one fold-1 round: no second concept on 1:1 codes, two answers for 4 of 11 multi-answer codes.
- 2026-10-13: added gpt-4o-mini (version 2024-07-18) to the model comparison (E10): expansion and choice on
  expanded Estonian with large and small embeddings, frozen prompts, provider-default temperature.
