"""Print every number reported in the paper, in reading order, from the files in outputs/ (no API calls).

  python analysis/paper_numbers.py

Needs the OMOP vocabulary in data/vocabulary/ (for the candidate set and the SNOMED CT hierarchy).
Agreement values are % of the 511 test codes; differences are percentage points (pp) with 95% CIs from a
bootstrap over NCSP groups (2,000 resamples, the same groups for both systems in paired comparisons).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import WORK_DIR  # noqa: E402
from evaluate import cluster_bootstrap_ci, paired_bootstrap, per_code_scores  # noqa: E402
from gold import load_gold_codes  # noqa: E402
from hierarchy import Hierarchy, relations_for_run  # noqa: E402
from results import MAIN, collect, rerank_predictions  # noqa: E402

ROUTES = {"et": "Estonian, unmodified", "et_translate": "Estonian, translated by LLM",
          "et_expand": "Estonian, expanded by LLM", "en": "English, unmodified", "en_expand": "English, expanded by LLM"}
USAGI = {"et_google": "Usagi, Google-translated", "et_translated": "Usagi, LLM-translated", "en": "Usagi, English name"}


def pct(x):
    return 100 * float(x)


def paired(a, b):
    p = paired_bootstrap(a, b)
    return f"{pct(p['diff']):.1f} pp (95% CI {pct(p['low']):.1f} to {pct(p['high']):.1f})"


def main():
    R = collect()
    gold = load_gold_codes()
    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
    test = gold[gold.source_code.isin(folds[folds.fold != "1"].source_code)].reset_index(drop=True)
    dev = gold[gold.source_code.isin(folds[folds.fold == "1"].source_code)]

    print("== Methods")
    print(f"codes per chapter: {gold.chapter.value_counts().reindex(list('LMT')).to_dict()} (total {len(gold)})")
    print(f"reference targets in the candidate set: {gold.mapping_type.value_counts().to_dict()}")
    print(f"development: {len(dev)} codes, {R['n_groups']['dev']} groups; test: {len(test)} codes, "
          f"{R['n_groups']['test']} groups")

    # Per-code scores: expanded inputs are per-code means over the three runs (as in Table 1).
    scores = {}
    for route in ROUTES:
        lang = route.split("_")[0]
        if route.endswith("_expand"):
            scores[route] = R["repeats"][lang]["mean_scores"]
        else:
            scores[route] = per_code_scores(rerank_predictions(f"{route}__{MAIN}"), test)

    print("\n== Table 1 (retrieval: reference concept in top 1/5/10/30; final prediction: agreement [95% CI], <=R1)")
    for route, label in ROUTES.items():
        se, rag = R["search"][route], R["rag"][route]
        print(f"{label:30s} {se['top1']:5.1f} {se['at5']:5.1f} {se['at10_mean']:5.1f} {se['at30']:5.1f}   "
              f"{rag['hit']:.1f} [{rag['hit_ci']}]   {rag['within_r1']:.1f}")
    usagi_scores = {}
    ids = set().union(*gold.index_targets)
    usagi_preds = {}
    for key in USAGI:
        u = pd.read_csv(WORK_DIR / "usagi" / f"usagi_output_{key}.csv", dtype=str, keep_default_na=False)
        u = u[u.sourceCode.isin(test.source_code)]
        usagi_preds[key] = dict(zip(u.sourceCode, u.conceptId))
        ids |= set(u.conceptId)
    hierarchy = Hierarchy.from_athena(ids)
    for key, label in USAGI.items():
        s = per_code_scores({c: [p] for c, p in usagi_preds[key].items()}, test)
        usagi_scores[key] = s
        low, high = cluster_bootstrap_ci(s, "hit")
        r1 = relations_for_run(usagi_preds[key], test, hierarchy).level.isin(["R0", "R1"]).mean()
        print(f"{label:30s}     -     -     -     -   {pct(s.hit.mean()):.1f} [{pct(low):.1f}–{pct(high):.1f}]   "
              f"{pct(r1):.1f}")
    print(f"expanded rows: three runs differed by at most {max(R['repeats'][l]['max_spread'] for l in ('et', 'en'))} pp")

    print("\n== Results: language gap and rewriting (paired differences)")
    print("English minus Estonian, unmodified:       ", paired(scores["en"], scores["et"]))
    print("translated minus unmodified Estonian:     ", paired(scores["et_translate"], scores["et"]),
          "(the paper reports 15.6, the difference of the rounded values)")
    print("expanded minus unmodified Estonian:       ", paired(scores["et_expand"], scores["et"]))
    print("translation minus expansion (Estonian):   ", paired(scores["et_translate"], scores["et_expand"]))
    print("English minus Estonian, both expanded:    ", paired(scores["en_expand"], scores["et_expand"]))

    print("\n== Results: retrieval and selection (expanded Estonian, means of three runs)")
    se, ce = R["search"]["et_expand"], R["repeats"]["et"]["choice_effect"]
    hit = R["rag"]["et_expand"]["hit"]
    no_ref, missed = 100 - se["at30"], se["at30"] - hit
    print(f"no reference concept among 30 candidates: {no_ref:.1f}%; reference available but not selected: "
          f"{missed:.1f}% -> {100 * missed / (missed + no_ref):.0f}% of nonmatching predictions despite retrieval")
    print(f"vs highest-ranked candidate: repaired {ce['repaired']}%, broken {ce['broken']}%; agreement "
          f"{se['top1']:.1f}% -> {hit:.1f}%")
    r1 = R["rag"]["et_expand"]["within_r1"]
    print(f"one step away: {100 * (r1 - hit) / (100 - hit):.0f}% of nonmatching predictions")
    print(f"reference concept among the top 5 candidates: {se['at5']:.1f}%")

    print("\n== Results: Usagi and other models")
    print("expanded-Estonian pipeline minus Usagi (Google):  ", paired(scores["et_expand"], usagi_scores["et_google"]))
    print("expanded-Estonian pipeline minus Usagi (English): ", paired(scores["et_expand"], usagi_scores["en"]))
    x = R["extra"]
    print(f"GPT-4o and/or small embeddings: {min(x['small_luna'], x['gpt4o_large_rag'], x['gpt4o_small_rag'])}–"
          f"{max(x['small_luna'], x['gpt4o_large_rag'], x['gpt4o_small_rag'])}%; "
          f"GPT-4o mini: {min(x['mini_large_rag'], x['mini_small_rag'])}–{max(x['mini_large_rag'], x['mini_small_rag'])}%")

    print("\n== Figure 1 (% of test codes: exact, 1, 2, 3 steps, further or unavailable)")
    hier, levels = R["hierarchy"], ["R0", "R1", "R2", "R3"]
    for route, label in ROUTES.items():
        if route.endswith("_expand"):
            m = R["repeats"][route.split("_")[0]]["levels_mean"]
            vals = [float(m.get(level, 0)) for level in levels]
            vals.append(100 - sum(vals))
        else:
            vals = [float(hier.loc[f"RAG | {route}", c]) for c in ["R0 exact", "R1", "R2", "R3", "further/none"]]
        print(f"{label:30s} " + "  ".join(f"{v:5.1f}" for v in vals))


if __name__ == "__main__":
    main()
