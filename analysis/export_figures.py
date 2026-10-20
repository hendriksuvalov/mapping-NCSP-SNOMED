"""Export the article figure as PDF (analysis/figures/), sized for the IOS Press type area (124 mm wide).

  python analysis/export_figures.py
"""
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from results import collect

OUT = Path(__file__).resolve().parent / "figures"
WIDTH = 12.4 * cm
FONT = 7  # IOS minimum: 8 pt in tables, 6 pt in other illustrations
# Times New Roman as shipped with macOS; set FONT_FILE to another TrueType Times font elsewhere.
FONT_FILE = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
pdfmetrics.registerFont(TTFont("TNR", str(FONT_FILE)))
pdfmetrics.registerFont(TTFont("TNR-B", str(FONT_FILE.with_name("Times New Roman Bold.ttf"))))

INK, GREY = colors.HexColor("#111111"), colors.HexColor("#666666")
# Ordinal blue ramp for R0-R3 and a neutral grey for "further or none" (readable in greyscale print).
LEVEL_COLORS = ["#104281", "#256abf", "#5598e7", "#86b6ef", "#D6D5D1"]
LEVEL_TEXT = ["#FFFFFF", "#FFFFFF", "#111111", "#111111", "#111111"]


def hierarchy_figure():
    """Stacked horizontal bars: share of test codes at each distance from the nearest reference concept, grouped by
    input language. Expanded inputs are means over three runs (as in Table 1), other inputs single runs."""
    R = collect()
    hier = R["hierarchy"]
    cols = ["R0 exact", "R1", "R2", "R3", "further/none"]
    levels = ["R0", "R1", "R2", "R3"]

    def single(key):  # one run, from the hierarchy summary table
        return [float(hier.loc[key, c]) for c in cols]

    def mean(lang):  # mean over the three runs, exact values (not reconstructed from rounded labels)
        shares = R["repeats"][lang]["levels_mean"]
        values = [float(shares.get(level, 0)) for level in levels]
        return values + [100 - sum(values)]

    groups = [("Estonian input", [("Unmodified", single("RAG | et")),
                                  ("LLM translation", single("RAG | et_translate")),
                                  ("Contextual expansion", mean("et"))]),
              ("English input", [("Unmodified", single("RAG | en")),
                                 ("Contextual expansion", mean("en"))])]
    names = ["Exact match", "1 step", "2 steps", "3 steps", "Further away or unavailable"]
    labels = [label for _, rows in groups for label, _ in rows] + [g for g, _ in groups]
    label_w = max(pdfmetrics.stringWidth(label, "TNR", FONT) for label in labels) + 10
    plot_w = WIDTH - label_w - 12  # room for the "100%" tick label
    bar_h, gap, head_h, group_gap, legend_h, axis_h = 7.5, 1.8, 8.5, 3, 12, 12
    n_rows = sum(len(rows) for _, rows in groups)
    h = legend_h + len(groups) * head_h + (len(groups) - 1) * group_gap + n_rows * (bar_h + gap) + axis_h
    d = Drawing(WIDTH, h)
    x = label_w
    for color, name in zip(LEVEL_COLORS, names):
        d.add(Rect(x, h - 9, 8, 8, fillColor=colors.HexColor(color), strokeColor=None))
        d.add(String(x + 11, h - 8, name, fontName="TNR", fontSize=FONT, fillColor=INK))
        x += 11 + pdfmetrics.stringWidth(name, "TNR", FONT) + 9
    y = h - legend_h
    for g, (heading, rows) in enumerate(groups):
        if g:
            y -= group_gap
        y -= head_h
        d.add(String(2, y + 2, heading, fontName="TNR-B", fontSize=FONT, fillColor=INK))
        for label, shares in rows:
            y -= bar_h + gap
            d.add(String(label_w - 5, y + 2.4, label, fontName="TNR", fontSize=FONT, fillColor=INK, textAnchor="end"))
            x = label_w
            for i, share in enumerate(shares):
                seg = plot_w * share / 100
                if seg <= 0:
                    continue
                d.add(Rect(x, y, max(seg - 1.2, 0.3), bar_h, fillColor=colors.HexColor(LEVEL_COLORS[i]), strokeColor=None))
                text = f"{share:.0f}"
                for size in (FONT, 6):  # 6 pt is the IOS minimum for lettering in figures
                    if seg - 1.2 >= pdfmetrics.stringWidth(text, "TNR", size) + 2:
                        d.add(String(x + (seg - 1.2) / 2, y + 2.4 + (FONT - size) / 2, text, fontName="TNR",
                                     fontSize=size, fillColor=colors.HexColor(LEVEL_TEXT[i]), textAnchor="middle"))
                        break
                else:
                    raise ValueError(f"no room for the label {text} in row {label}")
                x += seg
    for tick in (0, 25, 50, 75, 100):
        tx = label_w + plot_w * tick / 100
        d.add(Line(tx, y - 2, tx, y - 5, strokeColor=GREY, strokeWidth=0.5))
        d.add(String(tx, y - 11, f"{tick}%", fontName="TNR", fontSize=FONT, fillColor=GREY, textAnchor="middle"))
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(hierarchy_figure(), str(OUT / "hierarchy.pdf"))
    print(f"wrote {OUT / 'hierarchy.pdf'}")


if __name__ == "__main__":
    main()
