"""Write the Usagi input files (usagi/usagi_input_en.csv and usagi/usagi_input_et_translated.csv).

Both contain all 638 gold codes. The English file uses the official English NCSP names; the Estonian file uses the
same LLM translations the pipeline uses (queries/et_translate.csv), so every system gets identical input.
  python make_usagi_inputs.py
"""
from pathlib import Path

import pandas as pd

from gold import load_gold_codes
from config import WORK_DIR


def main():
    gold = load_gold_codes()
    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str).set_index("source_code").fold
    english = pd.read_csv(WORK_DIR / "queries/en.csv", dtype=str).set_index("source_code").text
    translated = pd.read_csv(WORK_DIR / "queries/et_translate.csv", dtype=str).set_index("source_code").text
    if not translated.index.is_unique or set(translated.index) != set(gold.source_code):
        raise SystemExit("queries/et_translate.csv must have exactly one translation per gold code")

    out = WORK_DIR / "usagi"
    out.mkdir(exist_ok=True)
    for filename, names in [("usagi_input_en.csv", english), ("usagi_input_et_translated.csv", translated)]:
        table = pd.DataFrame({
            "source_code": gold.source_code,
            "source_name": gold.source_code.map(names),
            "source_frequency": 1,
            "estonian_name": gold.description_et,
            "ncsp_chapter": gold.chapter,
            "set": gold.source_code.map(lambda c: "development" if folds[c] == "1" else "test"),
        })
        if table.source_name.isna().any() or table.source_name.str.strip().eq("").any():
            raise SystemExit(f"{filename}: empty source names")
        table.to_csv(out / filename, index=False)
        print(f"wrote {out / filename} ({len(table)} codes)")
        print(table.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
