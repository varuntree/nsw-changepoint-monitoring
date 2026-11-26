"""Fixture tests for the number lint: scripts/check_numbers.py.

Every mutation that an earlier form of the lint let pass (7 -> 8 with another 8 in the sentence, -0.18 -> -0.28 in
inline math, <0.001 -> <0.0001, "3 days" -> "14 days") must now fail, and so must short, missing and repeated anchors."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import check_numbers as C  # noqa: E402

FIX = HERE / "tests" / "fixtures"
RULES = C.load_whitelist()
COLS = ["id", "tex", "anchor", "printed", "value", "source", "select", "column", "transform", "rounding", "note"]


def row(id_, anchor, key, rounding="int", source="lint/vals.csv", column="value"):
    return (id_, "t.tex", anchor, "", "", source, f"key=='{key}'", column, "x", rounding, "")


def lint(tmp_path, text, rows=(), name="t.tex"):
    p = tmp_path / name
    p.write_text(text)
    led = pd.DataFrame([dict(zip(COLS, r)) for r in rows], columns=COLS)
    rep, *_ = C.run([p], [], led, RULES, results=FIX, check_fresh=False)
    return rep


HW = row("hw", "holt-winters with ewma catches", "hw_ewma")
OT = row("ot", "and the other monitor catches", "other")
GOOD = "Holt-Winters with EWMA catches 7 of the 15 episodes and the other monitor catches 8 of them."


def test_correct_text_passes(tmp_path):
    assert len(lint(tmp_path, GOOD, [HW, OT])) == 0


def test_changed_count_fails_even_if_the_value_is_elsewhere_in_the_sentence(tmp_path):
    rep = lint(tmp_path, GOOD.replace("catches 7", "catches 8"), [HW, OT])
    assert any(r.startswith("text 8, results give 7") for r in rep.reason)


def test_negative_number_in_inline_math(tmp_path):
    r = [row("c", "capped cusum loses with the same-weekday difference", "contrast", "dp2")]
    ok = "Capped CUSUM loses with the same-weekday difference ($-0.18$)."
    assert len(lint(tmp_path, ok, r)) == 0
    rep = lint(tmp_path, ok.replace("-0.18", "-0.28"), r)
    assert any("text -0.28, results give -0.18" in x for x in rep.reason)


def test_p_value_bound_in_inline_math(tmp_path):
    r = [row("p", "where chance alone would catch about one (p (own)", "p_own", "lt")]
    ok = "Where chance alone would catch about one (p (own) $<0.001$)."
    assert len(lint(tmp_path, ok, r)) == 0
    assert len(lint(tmp_path, ok.replace("0.001", "0.0001"), r))


def test_result_in_days_is_not_whitelisted(tmp_path):
    r = [row("m", "their median location error was", "median_error")]
    ok = "Their median location error was 3 days."
    assert len(lint(tmp_path, ok, r)) == 0
    assert len(lint(tmp_path, ok.replace("3 days", "14 days"), r))
    assert len(lint(tmp_path, "The median error was 3 days."))      # no row: a result in days is not a design constant


def test_short_anchor_fails(tmp_path):
    rep = lint(tmp_path, GOOD, [row("hw", "ewma catches", "hw_ewma"), OT])
    assert any("anchor shorter" in r for r in rep.reason)


def test_repeated_anchor_fails(tmp_path):
    text = GOOD + " Again, Holt-Winters with EWMA catches 7."
    assert any("occurs 2 times" in r for r in lint(tmp_path, text, [HW, OT]).reason)


def test_dead_row_fails(tmp_path):
    rep = lint(tmp_path, "Nothing here.", [HW])
    assert any("dead row" in r for r in rep.reason)


def test_anchor_must_end_directly_before_the_number(tmp_path):
    rep = lint(tmp_path, GOOD, [row("hw", "holt-winters with ewma", "hw_ewma"), OT])
    assert any("no number directly after" in r for r in rep.reason)


def test_two_rows_cannot_claim_one_number(tmp_path):
    rep = lint(tmp_path, GOOD, [HW, OT, row("hw2", "holt-winters with ewma catches", "hw_ewma")])
    assert any("occurs" in r or "two ledger rows" in r for r in rep.reason) or any("duplicate" in r for r in rep.reason)


def test_unclaimed_number_fails(tmp_path):
    rep = lint(tmp_path, GOOD, [HW])
    assert (rep.printed == "8").any()


def test_printed_precision(tmp_path):
    r = [("g", "t.tex", "with this forecast it gained", "", "", "lint/raw.csv", "", "gain", "x", "dp2", "")]
    assert len(lint(tmp_path, "With this forecast it gained 0.35 here.", r))
    assert len(lint(tmp_path, "With this forecast it gained 0.36 here.", r)) == 0


def test_year_passes(tmp_path):
    assert len(lint(tmp_path, "In 2025 nothing happened.")) == 0


def test_thousands_and_minus_and_ranges():
    toks = C.tokens("about 7{,}116 days, a fall of $-$0.18 and 60--76\\% of alarms")
    printed = [t["printed"] for t in toks if t["kind"] != "spelled"]
    assert "7116" in printed and "-0.18" in printed and "60" in printed and "76" in printed
    kinds = {t["printed"]: t["kind"] for t in toks}
    assert kinds["60"] == "range-lo" and kinds["76"] == "range-hi"


def test_inline_math_kept_or_blanked():
    s = C.remove_args(r"a $-0.18$ b $<0.001$ c $\le 0.006$ d $-17.7\%$ e $10^{-9}$ f $h=4$")
    printed = [t["printed"] for t in C.tokens(s)]
    assert printed == ["-0.18", "0.001", "0.006", "-17.7"]


def test_spelled_numbers_flagged(tmp_path):
    rep = lint(tmp_path, "Sixty-six fell on a weekend.")
    assert (rep.kind == "spelled").any()
    rep = lint(tmp_path, "Two alarms fell on a weekend.")
    assert (rep.kind == "spelled").any()


def test_hand_edited_generated_table_fails(tmp_path, monkeypatch):
    out = tmp_path / "tables"
    out.mkdir()
    (out / "x.tex").write_text("% written by scripts/tables.py from results/; do not edit\n\\caption{A}\n1 & 2 \\\\\n")

    def fake(tex_dir):
        want = "% written by scripts/tables.py from results/; do not edit\n\\caption{A}\n1 & 3 \\\\\n"
        rep = [] if (tex_dir / "x.tex").read_text() == want else [dict(file="tables/x.tex", line=0, token="", printed="",
                                                                         kind="table", context="", reason="generated table is stale")]
        return {"x.tex"}, rep
    monkeypatch.setattr(C, "fresh_tables", fake)
    rep, *_ = C.run([], [out / "x.tex"], pd.DataFrame(columns=COLS), RULES, check_fresh=True, tex_dir=out)
    assert any("stale" in r for r in rep.reason)
