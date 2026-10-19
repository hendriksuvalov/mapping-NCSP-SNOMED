"""Write the plain query files: queries/en.csv (official English NCSP names) and queries/et.csv (Estonian names).

Each query file has columns source_code, text. A code may have several rows (e.g. expansion variants);
retrieval then scores each concept by its best-matching row.
"""
from pathlib import Path

import pandas as pd

from config import GOLD_DIR, WORK_DIR
from gold import load_gold_codes


def main():
    codes = load_gold_codes()
    classifier = pd.read_csv(GOLD_DIR / "NCSP" / "ncsp_classifier.csv", dtype=str, keep_default_na=False)
    english = classifier.set_index("code").english_name
    missing = set(codes.source_code) - set(english.index)
    if missing:
        raise SystemExit(f"codes without an English name: {sorted(missing)}")

    out = WORK_DIR / "queries"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"source_code": codes.source_code, "text": codes.source_code.map(english)}).to_csv(out / "en.csv", index=False)
    pd.DataFrame({"source_code": codes.source_code, "text": codes.description_et}).to_csv(out / "et.csv", index=False)
    print(f"wrote {len(codes)} codes to {out}/en.csv and {out}/et.csv")


if __name__ == "__main__":
    main()
