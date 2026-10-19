"""Collect every number the article uses from the experiment result files (test codes, folds 2-5).

Missing runs give None. Run from the repository root or analysis/; the pipeline code is imported from pipeline/.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from evaluate import candidates_to_predictions, cluster_bootstrap_ci, mcnemar, paired_bootstrap, per_code_scores
from gold import load_gold_codes
from rerank import candidates_file
from config import WORK_DIR

ROUTES = ["et", "et_translate", "et_expand", "en", "en_expand"]
MAIN = "large-3072__k30__low__rerank.v2_f2345"


def pct(x):
    return None if x is None else round(100 * float(x), 1)


def ci_text(scores, metric="hit"):
    low, high = cluster_bootstrap_ci(scores, metric)
    return f"{100 * low:.1f}–{100 * high:.1f}"


def rerank_predictions(stem):
    path = WORK_DIR / "runs/rerank" / f"{stem}.csv"
    if not path.exists():
        return None
    run = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {c: [i] for c, i in zip(run.source_code, run.concept_id) if i}


def collect() -> dict:
    gold = load_gold_codes()
    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str)
    test = gold[gold.source_code.isin(folds[folds.fold != "1"].source_code)].reset_index(drop=True)
    R = {"n_test": len(test), "search": {}, "rag": {}, "no_search": {}, "extra": {}, "tests": {}}

    rag_scores = {}
    for route in ROUTES:
        cands = pd.read_csv(candidates_file(route, "large-3072"), dtype={"source_code": str, "concept_id": str})
        cands = cands[cands.source_code.isin(test.source_code)]
        top1 = per_code_scores(candidates_to_predictions(cands, 1), test)
        R["search"][route] = {
            "top1": pct(top1.hit.mean()), "top1_ci": ci_text(top1),
            **{f"at{k}": pct(per_code_scores(candidates_to_predictions(cands, k), test).hit.mean()) for k in (10, 30, 50)},
        }
        preds = rerank_predictions(f"{route}__{MAIN}")
        if preds is None:
            R["rag"][route] = None
            continue
        s = per_code_scores(preds, test)
        rag_scores[route] = s
        R["rag"][route] = {
            "hit": pct(s.hit.mean()), "hit_ci": ci_text(s), "complete": pct(s.complete.mean()),
            **{f"ch_{c}": pct(g.hit.mean()) for c, g in s.groupby("chapter")},
            **{f"type_{t}": pct(g.hit.mean()) for t, g in s.groupby("mapping_type")},
            **{f"complete_type_{t}": pct(g.complete.mean()) for t, g in s.groupby("mapping_type")},
        }

    hier_path = WORK_DIR / "runs/hierarchy/summary.csv"
    R["hierarchy"] = pd.read_csv(hier_path).set_index("run") if hier_path.exists() else None
    if R["hierarchy"] is not None:
        for route in ROUTES:
            key = f"RAG | {route}"
            if R["rag"].get(route) and key in R["hierarchy"].index:
                h = R["hierarchy"].loc[key]
                R["rag"][route].update(within_r1=h["within R1"], within_r2=h["within R2"],
                                       more_specific=h["near miss: more specific"],
                                       more_general=h["near miss: more general"], lateral=h["near miss: lateral"])

    for language in ("et", "en"):
        path = WORK_DIR / "runs/zero_shot" / f"{language}__low.csv"
        if not path.exists():
            R["no_search"][language] = None
            continue
        z = pd.read_csv(path, dtype=str, keep_default_na=False)
        z = z[z.source_code.isin(test.source_code)]  # the run covers all 638 codes; report the test codes
        by_id = per_code_scores({c: [i] for c, i in zip(z.source_code, z.concept_id) if i}, test)
        by_name = per_code_scores({c: [i] for c, i in zip(z.source_code, z.concept_id_by_name) if i}, test)
        R["no_search"][language] = {"by_id": pct(by_id.hit.mean()), "by_name": pct(by_name.hit.mean())}

    for key, stem in {"k10": "et_expand__large-3072__k10__low__rerank.v2_f2345",
                      "k50": "et_expand__large-3072__k50__low__rerank.v2_f2345",
                      "small_luna": "et_expand__small-1536__k30__low__rerank.v2_f2345"}.items():
        preds = rerank_predictions(stem)
        R["extra"][key] = None if preds is None else pct(per_code_scores(preds, test).hit.mean())
    for key, route, variant in [("small", "et_expand", "small-1536"), ("gpt4o_large", "et_expand_gpt4o", "large-3072"),
                                ("gpt4o_small", "et_expand_gpt4o", "small-1536"),
                                ("mini_large", "et_expand_gpt4omini", "large-3072"),
                                ("mini_small", "et_expand_gpt4omini", "small-1536")]:
        path = candidates_file(route, variant)
        if not path.exists():
            R["extra"][f"{key}_top1"] = R["extra"][f"{key}_at30"] = None
            continue
        cands = pd.read_csv(path, dtype={"source_code": str, "concept_id": str})
        cands = cands[cands.source_code.isin(test.source_code)]
        R["extra"][f"{key}_top1"] = pct(per_code_scores(candidates_to_predictions(cands, 1), test).hit.mean())
        R["extra"][f"{key}_at30"] = pct(per_code_scores(candidates_to_predictions(cands, 30), test).hit.mean())
    for key, stem in {"gpt4o_large_rag": "et_expand_gpt4o__large-3072__k30__gpt4o__rerank.v2_f2345",
                      "gpt4o_small_rag": "et_expand_gpt4o__small-1536__k30__gpt4o__rerank.v2_f2345",
                      "mini_large_rag": "et_expand_gpt4omini__large-3072__k30__gpt4omini__rerank.v2_f2345",
                      "mini_small_rag": "et_expand_gpt4omini__small-1536__k30__gpt4omini__rerank.v2_f2345"}.items():
        preds = rerank_predictions(stem)
        R["extra"][key] = None if preds is None else pct(per_code_scores(preds, test).hit.mean())

    pairs = [("et_expand", "en_expand"), ("et_expand", "et"), ("et_expand", "et_translate"), ("et", "en")]
    for a, b in pairs:
        if a in rag_scores and b in rag_scores:
            m = mcnemar(rag_scores[a], rag_scores[b])
            R["tests"][f"{a}_vs_{b}"] = m

    # Repeats (E3): expanded inputs were run three times end to end; report mean and range for those rows.
    from hierarchy import Hierarchy, relations_for_run
    R["repeats"] = {}
    repeat_runs = {}
    for lang in ("et", "en"):
        routes = [f"{lang}_expand", f"{lang}_expand_r2", f"{lang}_expand_r3"]
        stems = [f"{r}__{MAIN}" for r in routes]
        if all((WORK_DIR / "runs/rerank" / f"{s}.csv").exists() for s in stems):
            repeat_runs[lang] = (routes, [rerank_predictions(s) for s in stems])
    if repeat_runs:
        ids = set().union(*gold.index_targets) | {p[0] for _, runs in repeat_runs.values() for run in runs for p in run.values()}
        for routes, _ in repeat_runs.values():  # top search results too (retrieval-only row of the hierarchy figure)
            for route in routes:
                c = pd.read_csv(candidates_file(route, "large-3072"), dtype={"source_code": str, "concept_id": str})
                ids |= set(c.loc[c["rank"] == 1, "concept_id"])
        hierarchy = Hierarchy.from_athena(ids)
    for lang, (routes, runs) in repeat_runs.items():
        per_run = []
        for route, preds in zip(routes, runs):
            s = per_code_scores(preds, test)
            rel = relations_for_run({c: p[0] for c, p in preds.items()}, test, hierarchy)
            near = rel.distance.isin([1, 2])
            cands = pd.read_csv(candidates_file(route, "large-3072"), dtype={"source_code": str, "concept_id": str})
            cands = cands[cands.source_code.isin(test.source_code)]
            top1 = candidates_to_predictions(cands, 1)
            top1_rel = relations_for_run({c: p[0] for c, p in top1.items()}, test, hierarchy)
            per_run.append({
                "levels": 100 * rel.level.value_counts(normalize=True),          # share of test codes per R level
                "levels_search": 100 * top1_rel.level.value_counts(normalize=True),
                "hit": 100 * s.hit.mean(),
                "within_r1": 100 * rel.level.isin(["R0", "R1"]).mean(),
                "within_r2": 100 * rel.level.isin(["R0", "R1", "R2"]).mean(),
                "more_specific": 100 * (near & (rel.kind == "more_specific")).mean(),
                "more_general": 100 * (near & (rel.kind == "more_general")).mean(),
                **{f"at{k}": 100 * per_code_scores(candidates_to_predictions(cands, k), test).hit.mean() for k in (1, 10, 30, 50)},
                "scores": s,
                "top1_scores": per_code_scores(candidates_to_predictions(cands, 1), test),
            })
        stats = {}
        for key in ["hit", "within_r1", "within_r2", "more_specific", "more_general", "at1", "at10", "at30", "at50"]:
            values = [r[key] for r in per_run]
            stats[key] = (round(sum(values) / len(values), 1), round(min(values), 1), round(max(values), 1))
        R["repeats"][lang] = stats
        R["repeats"][lang]["levels_mean"] = sum(r["levels"] for r in per_run).fillna(0) / len(per_run)
        R["repeats"][lang]["levels_search_mean"] = sum(r["levels_search"] for r in per_run).fillna(0) / len(per_run)
        base = f"{lang}_expand"
        R["rag"][base]["hit_run1"] = R["rag"][base]["hit"]
        for key in ["hit", "within_r1", "within_r2", "more_specific", "more_general"]:
            R["rag"][base][key] = stats[key][0]
        # 95% CI of the three-run mean: average each code's outcome over the runs, then resample NCSP groups.
        def mean_over_runs(frames):
            merged = frames[0][["source_code", "group"]].copy()
            merged["hit"] = sum(f.set_index("source_code").loc[merged.source_code, "hit"].to_numpy() for f in frames) / len(frames)
            return merged
        rag_mean = mean_over_runs([r["scores"] for r in per_run])
        top1_mean = mean_over_runs([r["top1_scores"] for r in per_run])
        R["rag"][base]["hit_ci"] = ci_text(rag_mean)
        R["search"][base]["run1"] = {k: R["search"][base][k] for k in ("top1", "at10", "at30", "at50")}
        R["search"][base].update(top1=stats["at1"][0], top1_ci=ci_text(top1_mean),
                                 at10=stats["at10"][0], at30=stats["at30"][0], at50=stats["at50"][0])
        R["repeats"][lang]["max_spread"] = round(stats["hit"][2] - stats["hit"][1], 1)
        R["repeats"][lang]["runs"] = per_run
        R["repeats"][lang]["mean_scores"] = rag_mean
        types = test.set_index("source_code").mapping_type
        R["rag"][base]["type11_main"] = pct(rag_mean.hit[rag_mean.source_code.map(types).eq("1:1").to_numpy()].mean())
        R["rag"][base]["type1n_main"] = pct(rag_mean.hit[rag_mean.source_code.map(types).eq("1:n").to_numpy()].mean())
        # What the LLM's choice does to the top search result (mean over the three runs, % of test codes).
        effects = []
        for r in per_run:
            rag_hit = r["scores"].set_index("source_code").hit
            top_hit = r["top1_scores"].set_index("source_code").hit.reindex(rag_hit.index)
            effects.append({"kept": ((top_hit == 1) & (rag_hit == 1)).mean(), "repaired": ((top_hit == 0) & (rag_hit == 1)).mean(),
                            "broken": ((top_hit == 1) & (rag_hit == 0)).mean(),
                            "selection_success": rag_hit.mean() / (r["at30"] / 100)})
        R["repeats"][lang]["choice_effect"] = {k: round(100 * sum(e[k] for e in effects) / len(effects), 1) for k in effects[0]}
    if len(repeat_runs) == 2:
        pvals = [mcnemar(a["scores"], b["scores"])["p_value"]
                 for a, b in zip(R["repeats"]["et"]["runs"], R["repeats"]["en"]["runs"])]
        R["tests"]["repeat_p_max"] = max(pvals)

    for entry in R["rag"].values():  # main-text subset values: three-run means where available, else the single run
        if entry:
            entry.setdefault("type11_main", entry.get("type_1:1"))
            entry.setdefault("type1n_main", entry.get("type_1:n"))

    # Primary comparisons (paired, NCSP groups resampled jointly); expanded inputs use the per-code mean over runs.
    def frame_for(route):
        lang = route.split("_")[0]
        if route.endswith("_expand") and lang in R["repeats"] and "mean_scores" in R["repeats"][lang]:
            return R["repeats"][lang]["mean_scores"]
        return rag_scores.get(route)
    R["paired"] = {}
    for name, (a_route, b_route) in {"rewriting": ("et_expand", "et"), "language": ("et_expand", "en_expand"),
                                     "translate_vs_expand": ("et_translate", "et_expand")}.items():
        fa, fb = frame_for(a_route), frame_for(b_route)
        if fa is not None and fb is not None:
            res = paired_bootstrap(fa, fb)
            R["paired"][name] = {k: (round(100 * v, 1) if k in ("diff", "low", "high") else v) for k, v in res.items()}
    groups = gold.assign(test=gold.source_code.isin(test.source_code))
    R["n_groups"] = {"test": int(groups[groups.test].group.nunique()), "dev": int(groups[~groups.test].group.nunique())}

    # Appendix A5: variant that may return up to two concepts (answers stored as "a;b").
    R["multi"] = {}
    for lang in ("et", "en"):
        path = WORK_DIR / "runs/rerank" / f"{lang}_expand__large-3072__k30__low__rerank_multi.v1_f2345.csv"
        if not path.exists():
            R["multi"][lang] = None
            continue
        run = pd.read_csv(path, dtype=str, keep_default_na=False)
        preds = {c: [x for x in i.split(";") if x] for c, i in zip(run.source_code, run.concept_id) if i}
        s = per_code_scores(preds, test).set_index("source_code")
        two = pd.Series({c: len(p) == 2 for c, p in preds.items()}).reindex(s.index, fill_value=False)
        one_to_n = s.mapping_type == "1:n"
        R["multi"][lang] = {
            "hit": pct(s.hit.mean()), "complete": pct(s.complete.mean()), "precision": pct(s.precision.mean()),
            "two_share": pct(two.mean()), "two_on_1to1": pct(two[s.mapping_type == "1:1"].mean()),
            "two_on_1ton": pct(two[one_to_n].mean()), "complete_1ton": pct(s.complete[one_to_n].mean()),
        }

    # Recall curve for the figure: share of test codes with a correct concept among the top k search candidates,
    # k = 1..50; expanded inputs averaged over their three runs.
    curve_routes = {"et": ["et"], "et_translate": ["et_translate"],
                    "et_expand": ["et_expand", "et_expand_r2", "et_expand_r3"],
                    "en": ["en"], "en_expand": ["en_expand", "en_expand_r2", "en_expand_r3"]}
    R["curve"] = {}
    for key, routes in curve_routes.items():
        frames = []
        for route in routes:
            path = candidates_file(route, "large-3072")
            if path.exists():
                frame = pd.read_csv(path, dtype={"source_code": str, "concept_id": str})
                frames.append(frame[frame.source_code.isin(test.source_code) & (frame["rank"] <= 50)])
        if not frames:
            continue
        values = []
        for k in range(1, 51):
            values.append(sum(per_code_scores(candidates_to_predictions(f, k), test).hit.mean() for f in frames)
                          / len(frames) * 100)
        R["curve"][key] = values
    for key, values in R["curve"].items():
        R["search"].setdefault(key, {}).update(at5=round(values[4], 1), at10_mean=round(values[9], 1))

    tokens = {}
    for name, path in {"rerank": WORK_DIR / "runs/rerank" / f"et_expand__{MAIN}.json",
                       "expand": WORK_DIR / "queries/et_expand.json", "translate": WORK_DIR / "queries/et_translate.json",
                       "no_search": WORK_DIR / "runs/zero_shot/et__low.json"}.items():
        if path.exists():
            m = json.loads(path.read_text())
            tokens[name] = (round(m["prompt_tokens"] / m["n_codes"]), round(m["completion_tokens"] / m["n_codes"]))
    R["tokens"] = tokens
    return R
