"""Google Translate route for the Usagi baseline.

Step 1 (no arguments): write usagi/google/estonian_terms.txt, one Estonian NCSP name per line, in a fixed order.
Someone translates that file with Google Translate (translate.google.com → Documents, Estonian → English) and saves
the result as usagi/google/english_terms.txt (one line per term, same order).

Step 2 (--build): check the translated file line by line and write usagi/usagi_input_et_google.csv in the same
format as the other Usagi inputs.

  python make_usagi_google_input.py            # step 1
  python make_usagi_google_input.py --build    # step 2
"""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from gold import load_gold_codes
from config import WORK_DIR

FOLDER = WORK_DIR / "usagi/google"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="build the Usagi input from the translated file")
    args = parser.parse_args()

    gold = load_gold_codes().sort_values("source_code").reset_index(drop=True)
    terms = gold.description_et.str.replace(r"\s+", " ", regex=True).str.strip()
    FOLDER.mkdir(parents=True, exist_ok=True)

    if not args.build:
        (FOLDER / "estonian_terms.txt").write_text("\n".join(terms) + "\n", encoding="utf-8")
        gold[["source_code"]].assign(estonian_name=terms).to_csv(FOLDER / "line_order.csv", index=False)
        print(f"wrote {FOLDER / 'estonian_terms.txt'} ({len(terms)} lines); translate it into english_terms.txt")
        return

    translated_path = FOLDER / "english_terms.txt"
    lines = [line.strip() for line in translated_path.read_text(encoding="utf-8").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) != len(terms):
        raise SystemExit(f"{translated_path} has {len(lines)} lines, expected {len(terms)}; lines must match one to one")
    if any(not line for line in lines):
        raise SystemExit(f"{translated_path} has empty lines; each Estonian term needs exactly one translation")

    folds = pd.read_csv(WORK_DIR / "splits/folds.csv", dtype=str).set_index("source_code").fold
    table = pd.DataFrame({
        "source_code": gold.source_code,
        "source_name": lines,
        "source_frequency": 1,
        "estonian_name": terms,
        "ncsp_chapter": gold.chapter,
        "set": gold.source_code.map(lambda c: "development" if folds[c] == "1" else "test"),
    })
    table.to_csv(WORK_DIR / "usagi/usagi_input_et_google.csv", index=False)
    (FOLDER / "TRANSLATED_ON.txt").write_text(f"Google Translate output read on {date.today()}\n")
    print(f"wrote usagi/usagi_input_et_google.csv ({len(table)} codes). Spot check:")
    print(table[["source_code", "estonian_name", "source_name"]].sample(8, random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()
