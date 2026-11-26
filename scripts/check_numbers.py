"""Number lint: every number printed in the thesis must come from the results.

Inputs   thesis/chapters/*.tex, thesis/tables/*.tex; ledger results/extra/quoted.csv (written only by
         scripts/numbers_extra.py); whitelist check_numbers_whitelist.csv (pattern, scope, reason).
Checks
  1. generated tables are fresh: tables.py is run into a temporary directory and every file it writes must equal the
     file in thesis/tables (a table it writes, or whose first line is "% written by scripts/tables.py" or
     "...recommend.py", is "generated"); only their \\caption{...} and \\multicolumn{..}{p{..}}{...} text is linted;
  2. every ledger row (one row = one printed number) names its file and an anchor: at least four words of the text
     that occur exactly once in the file and end directly before the number (only spaces, brackets, $, ~ or other
     non-alphanumeric characters between). The row's value is recomputed from source / select / column / transform /
     rounding and compared with the number actually printed there. A row fails if its anchor is too short, missing
     (dead row: the sentence is gone), not unique, not followed directly by a number, shares its number with another
     row, or if the printed number differs from the recomputed one (no comparison with a stored value);
  3. every number token not claimed by a ledger row must pass a whitelist rule (non-results only), else it fails;
  4. inline math is linted when it is a signed number, a bound or a percentage ($-0.18$, $<0.001$, $\\le 0.006$,
     $-17.7\\%$) or an assignment of a decimal ($\\hat\\phi = 0.40$); other math is blanked;
  5. spelled numbers: every spelled number from eleven up, and one to ten within three words of a counted noun.
Output   results/check_numbers_report.csv (file, line, token, context, reason) and a summary line
         "N tokens, M quoted, W whitelisted, F failures"; exit 1 on any failure.
Usage    python scripts/check_numbers.py [--no-fresh]"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
THESIS = HERE / "thesis"   # the thesis source is not in this repository
RES = HERE / "results"
LEDGER = RES / "extra" / "quoted.csv"
WHITELIST = HERE / "check_numbers_whitelist.csv"
REPORT = RES / "check_numbers_report.csv"
GEN_HEADERS = ("% written by scripts/tables.py", "% written by scripts/recommend.py")

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# "one" is left out of the counted-noun rule: it is an article or pronoun far more often than a count (a documented
# choice); "eleven" and up are always flagged
SPELLED_UNITS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
TEENS = ["eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
COUNTED = r"alarms?|episodes?|monitors?|boundar(?:y|ies)|events?|days?|weeks?|months?|years?|times?|per cent"

# ------------------------------------------------------------------------------------------------ extraction
DROP_ARG = r"\\(?:label|ref|eqref|autoref|pageref|cref|Cref|input|include|includegraphics|url|setlength|hspace\*?|vspace\*?|rule|bibliography|bibliographystyle|addcontentsline|newcommand|renewcommand|definecolor|linespread|arraystretch)"
CITE = r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)\*?"


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def blank(s, a, b):
    """Replace s[a:b] by spaces, keeping newlines (so line numbers survive)."""
    return s[:a] + re.sub(r"[^\n]", " ", s[a:b]) + s[b:]


def _match_brace(s, i):
    """i at '{'; return index after the matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def remove_args(s):
    # optional [..] and the brace arguments of dropped commands
    for pat in (DROP_ARG, CITE, r"\\href"):
        for m in list(re.finditer(pat, s))[::-1]:
            j = m.end()
            while j < len(s) and s[j] in " [":
                if s[j] == "[":
                    j = s.find("]", j) + 1
                else:
                    j += 1
            n_args = {"\\setlength": 2, "\\rule": 2, "\\newcommand": 2, "\\renewcommand": 2, "\\definecolor": 3}.get(
                re.sub(r"\*$", "", m.group(0)), 1)
            k = j
            for _ in range(n_args):
                while k < len(s) and s[k] == " ":
                    k += 1
                if k < len(s) and s[k] == "{":
                    k = _match_brace(s, k)
            s = blank(s, m.start(), k)
    # environments that hold constants, and tabular column specs
    for env in ("equation", "equation\\*", "align", "align\\*", "gather", "gather\\*"):
        for m in list(re.finditer(rf"\\begin{{{env}}}.*?\\end{{{env}}}", s, flags=re.S))[::-1]:
            s = blank(s, m.start(), m.end())
    for m in list(re.finditer(r"\\begin{(?:tabular\*?|longtable|tabularx)}(\[[^\]]*\])?(\{[^{}]*\})?", s))[::-1]:
        j = m.end()
        if j < len(s) and s[j] == "{":
            j = _match_brace(s, j)
        s = blank(s, m.start(), j)
    for m in list(re.finditer(r"\\begin{\w+\*?}(\[[^\]]*\])?", s))[::-1]:
        s = blank(s, m.start(), m.end())
    for m in list(re.finditer(r"\\(?:multicolumn|multirow)\{\d+\}\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}", s))[::-1]:
        s = blank(s, m.start(), m.end())
    for m in list(re.finditer(r"p\{[^{}]*\}|\[width=[^\]]*\]|\\linewidth|\\textwidth|-?\d+(?:\.\d+)?(?:pt|cm|mm|em|ex|in)\b", s))[::-1]:
        s = blank(s, m.start(), m.end())
    for m in list(re.finditer(r"\\begin{description}.*?\\end{description}", s, flags=re.S))[::-1]:
        block = s[m.start():m.end()]
        block = re.sub(r"\\item\[[^\]]*\]", lambda x: " " * len(x.group(0)), block)
        s = s[:m.start()] + block + s[m.end():]
    # inline math: mathematical constants and indices are not results; keep only an assignment of a decimal value
    # ($\hat\phi = 0.40$) and the sign/operator snippets used in prose ($-$, $<$, $\times$, $\mu$)
    def _math(mm):
        body = mm.group(0)
        if re.fullmatch(r"\$(-|<|>|\\le|\\ge|\\times|\\mu|\\pm|\\approx|\\sim)\$", body):
            return body
        if re.search(r"=\s*-?\d+\.\d+\s*\$$", body):
            return body
        # a p-value bound written with its symbol ($p<0.001$): the number is a result
        if re.fullmatch(r"\$\s*p\s*(?:<|>|\\le|\\ge|\\leq|\\geq)\s*\d+(?:\.\d+)?\s*\$", body):
            return body
        # signed numbers, bounds and percentages are results
        if re.fullmatch(r"\$\s*(?:<|>|\\le|\\ge|\\leq|\\geq|=)?\s*-?\d+(?:\.\d+)?\s*(?:\\%)?\s*\$", body):
            return body
        return " " * len(body)
    s = re.sub(r"(?<!\\)\$[^$]+\$", _math, s)
    # spacing macros that carry digits
    s = re.sub(r"\\(?:cmidrule|cline)(\([^)]*\))?\{[^}]*\}", lambda x: " " * len(x.group(0)), s)
    s = re.sub(r"\\(?:addlinespace|vskip|hskip|kern)\s*(\[[^\]]*\])?", lambda x: " " * len(x.group(0)), s)
    s = re.sub(r"\\(?:footnotesize|small|tiny|scriptsize|tabcolsep|arraystretch)\b", lambda x: " " * len(x.group(0)), s)
    return s


def caption_only(s):
    """Keep only caption text and p{} footnote cells of a generated table (blank the rest)."""
    keep = np.zeros(len(s), bool)
    for m in re.finditer(r"\\caption\{", s):
        keep[m.start():_match_brace(s, m.end() - 1)] = True
    for m in re.finditer(r"\\multicolumn\{\d+\}\{[^{}]*p\{[^{}]*\}[^{}]*\}\{", s):
        keep[m.start():_match_brace(s, m.end() - 1)] = True
    return "".join(c if (k or c == "\n") else " " for c, k in zip(s, keep))


NUM = re.compile(r"""(?<![\w\\.])
    (?P<neg>\$-\$|−|(?<![-\d\w])-)?
    (?P<num>\d{1,3}(?:\{,\}\d{3})+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<unit>\\%|\\,ms|\s?µs|\s?\$\\mu\$s|\s?SD|\s?times)?
    """, re.X)


def normalise(num, neg=None):
    v = num.replace("{,}", "").replace(",", "")
    return ("-" if neg else "") + v


def sentences(text):
    """(start, end) spans of sentences: split at '. ' / '.\\n' / ';' / blank lines."""
    spans, a = [], 0
    for m in re.finditer(r"(?<!\d)\.(?=\s)|(?<=\d)\.(?=\s+[A-Z\\])|;|\n\s*\n", text):
        spans.append((a, m.end()))
        a = m.end()
    spans.append((a, len(text)))
    return spans


def tokens(text):
    out = []
    for m in NUM.finditer(text):
        num = m.group("num")
        out.append(dict(start=m.start(), end=m.end(), printed=normalise(num, m.group("neg")), raw=m.group(0),
                        unit=(m.group("unit") or "").strip(), kind="number"))
    # ranges: a--b or "a to b" -> mark
    for i in range(len(out) - 1):
        gap = text[out[i]["end"]:out[i + 1]["start"]]
        if re.fullmatch(r"\s*(--|–|to)\s*", gap):
            out[i]["kind"], out[i + 1]["kind"] = "range-lo", "range-hi"
    # spelled numbers
    big = r"\b(?:(?:%s)(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|%s)\b" % ("|".join(TENS), "|".join(TEENS))
    for m in re.finditer(big, text, flags=re.I):
        out.append(dict(start=m.start(), end=m.end(), printed=m.group(0).lower(), raw=m.group(0), unit="", kind="spelled"))
    small = r"\b(%s)\b(?=(?:\W+\w+){0,3}?\W+(?:%s)\b)" % ("|".join(SPELLED_UNITS), COUNTED)
    for m in re.finditer(small, text, flags=re.I):
        out.append(dict(start=m.start(), end=m.end(), printed=m.group(0).lower(), raw=m.group(0), unit="", kind="spelled"))
    return out


# ------------------------------------------------------------------------------------------------ recomputation
def _round(v, rounding):
    if rounding == "int":
        return f"{v:.0f}" if abs(v - round(v)) != 0.5 else str(int(np.floor(v + 0.5)))
    if rounding in ("dp1", "dp2", "dp3", "dp4"):
        d = int(rounding[2])
        q = np.floor(abs(v) * 10 ** d + 0.5 + 1e-9) / 10 ** d
        return f"{np.sign(v) * q:.{d}f}".replace("-0.", "0.") if q == 0 else f"{np.sign(v) * q:.{d}f}"
    if rounding == "sig2":
        return f"{float(f'{v:.2g}'):g}"
    if rounding == "thousands":
        return f"{int(np.floor(v + 0.5))}"
    if rounding == "hundreds":
        return f"{int(np.floor(v / 100 + 0.5) * 100)}"
    if rounding == "lt":
        return "0.001" if v < 0.001 else f"{v:.3f}"
    if rounding.startswith("gtw"):           # e.g. gtw5: printed as the lower bound "more than five" when v > 5
        thr = int(rounding[3:])
        return _round(float(thr), "words") if v > thr else _round(v, "words")
    if rounding.startswith("lt"):            # e.g. lt0.002: printed as the bound "<0.002" below it (Monte Carlo floor)
        thr = rounding[2:]
        return thr if v < float(thr) else f"{v:.{max(1, len(thr.split('.')[-1]) if '.' in thr else 0)}f}"
    if rounding == "str":
        return str(v)
    if rounding == "words":
        n = int(np.floor(v + 0.5))
        small = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"] + TEENS
        if n < 20:
            return small[n]
        t, u = divmod(n, 10)
        return TENS[t - 2] + ("" if u == 0 else "-" + small[u])
    raise ValueError(rounding)


_CACHE = {}


def recompute(row, results=RES, cache=_CACHE):
    src = (results / row["source"]).resolve()
    if src not in cache:
        cache[src] = pd.read_csv(src)
    df = cache[src]
    if isinstance(row["select"], str) and row["select"].strip():
        df = df.query(row["select"].replace("&#44;", ","), engine="python")
    col, tr = row["column"], str(row["transform"])
    vals = df[col]
    agg, _, scale = tr.partition("*")          # e.g. "max*100": aggregate, then scale
    if agg in ("min", "max", "median", "count", "mean", "sum", "nunique"):
        v = float(getattr(vals, agg)()) if agg != "count" else float(len(vals))
        v = v * float(scale) if scale else v
    else:
        if len(vals) != 1:
            raise ValueError(f"select gives {len(vals)} rows")
        v = vals.iloc[0]
        if tr == "x":
            pass
        elif tr == "100x":
            v = 100 * float(v)
        elif tr == "1000x":
            v = 1000 * float(v)
        elif tr == "abs":
            v = abs(float(v))
        elif tr == "x/365*100":
            v = float(v) / 365 * 100
        elif tr == "neg":
            v = -float(v)
        else:
            raise ValueError(tr)
    if str(row["rounding"]) == "str":
        return str(v)
    return _round(float(v), str(row["rounding"]))


def norm_printed(p):
    p = str(p).strip()
    if re.fullmatch(r"-?[\d,{}.]+", p):
        neg = p.startswith("-")
        return normalise(p.lstrip("-"), neg)
    return p.lower()


# ------------------------------------------------------------------------------------------------ main check
def _split_top(pattern):
    """Split a regex at its top-level '|' so that overlapping alternatives are each searched (finditer would
    otherwise consume 'rounded to the nearest' and never see 'nearest 100')."""
    parts, depth, cur, esc = [], 0, "", False
    for ch in pattern:
        if esc:
            cur += ch
            esc = False
            continue
        if ch == "\\":
            cur += ch
            esc = True
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def load_whitelist(path=WHITELIST):
    w = pd.read_csv(path)
    return [(re.compile(alt, flags=re.I), str(r.scope), r.reason) for r in w.itertuples() for alt in _split_top(r.pattern)]


def whitelisted(text, tok, rules, fname):
    for pat, scope, reason in rules:
        if scope not in ("all", "") and not re.search(scope, fname):
            continue
        for m in pat.finditer(text, max(0, tok["start"] - 80), min(len(text), tok["end"] + 80)):
            if m.start() <= tok["start"] and tok["end"] <= m.end():
                return reason
    return None


def norm_map(text):
    """Lower-case text with runs of whitespace collapsed to one space, and each character's index in text."""
    out, idx, sp = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if sp:
                continue
            out.append(" ")
            sp = True
        else:
            out.append(ch.lower())
            sp = False
        idx.append(i)
    return "".join(out), idx


def norm_anchor(a):
    return re.sub(r"\s+", " ", str(a).strip().lower())


def anchor_words(a):
    return len(re.findall(r"[A-Za-z0-9]+", str(a)))


MIN_ANCHOR_WORDS = 4


def claim(row, text, nt, idx, toks):
    """Return (token, None) for the number directly after the row's anchor, or (None, reason)."""
    a = norm_anchor(row["anchor"])
    if anchor_words(a) < MIN_ANCHOR_WORDS:
        return None, f"anchor shorter than {MIN_ANCHOR_WORDS} words"
    hits = [m.start() for m in re.finditer("(?=" + re.escape(a) + ")", nt)]
    if not hits:
        return None, "dead row: anchor not in the text"
    if len(hits) > 1:
        return None, f"anchor occurs {len(hits)} times"
    end = idx[hits[0] + len(a) - 1] + 1
    nxt = [t for t in toks if t["start"] >= end]
    if not nxt:
        return None, "no number directly after the anchor"
    t = min(nxt, key=lambda t: t["start"])
    if re.search(r"[A-Za-z0-9]", text[end:t["start"]]):
        return None, "no number directly after the anchor"
    return t, None


def lint_file(fname, text, rows, rules, results):
    """Ledger rows of this file claim their numbers; unclaimed tokens must be whitelisted. Returns (report, q, w)."""
    rep, claimed = [], {}
    toks = tokens(text)
    nt, idx = norm_map(text)

    def line(pos):
        return text.count("\n", 0, pos) + 1

    def ctx(pos, end):
        return re.sub(r"\s+", " ", text[max(0, pos - 60):end + 60])

    for _, row in rows.iterrows():
        t, why = claim(row, text, nt, idx, toks)
        if why:
            rep.append(dict(file=fname, line=0, token="", printed="", kind="ledger", context=f"{row['id']}: {row['anchor']}",
                            reason=why))
            continue
        key = (t["start"], t["end"])
        if key in claimed:
            rep.append(dict(file=fname, line=line(t["start"]), token=t["raw"].strip(), printed=t["printed"], kind="ledger",
                            context=f"{row['id']} and {claimed[key]}", reason="two ledger rows claim the same number"))
            continue
        claimed[key] = row["id"]
        try:
            got = recompute(row, results)
        except Exception as e:  # noqa: BLE001
            rep.append(dict(file=fname, line=line(t["start"]), token=t["raw"].strip(), printed=t["printed"], kind="ledger",
                            context=str(row["id"]), reason=f"cannot recompute: {e!r}"[:200]))
            continue
        if norm_printed(got) != t["printed"]:
            rep.append(dict(file=fname, line=line(t["start"]), token=t["raw"].strip(), printed=t["printed"], kind="ledger",
                            context=f"{row['id']}: {ctx(t['start'], t['end'])}",
                            reason=f"text {t['printed']}, results give {norm_printed(got)}"))
    n_w = 0
    for tok in toks:
        if (tok["start"], tok["end"]) in claimed:
            continue
        reason = whitelisted(text, tok, rules, fname)
        if reason:
            n_w += 1
            continue
        rep.append(dict(file=fname, line=line(tok["start"]), token=tok["raw"].strip(), printed=tok["printed"], kind=tok["kind"],
                        context=ctx(tok["start"], tok["end"]), reason="no ledger row or whitelist rule"))
    return rep, len(claimed), n_w


def fresh_tables(tex_dir):
    """Run tables.py and recommend's table into a temp dir; return (generated names, report rows for stale files)."""
    sys.path.insert(0, str(HERE / "scripts"))
    rep, names = [], set()
    with tempfile.TemporaryDirectory() as tmp:
        tables = importlib.import_module("tables")
        tables.OUT = Path(tmp)
        for f in tables.ALL:
            try:
                f()
            except Exception as e:  # noqa: BLE001
                rep.append(dict(file=f.__name__, line=0, token="", printed="", kind="table", context=repr(e)[:200],
                                reason="tables.py function failed"))
        for p in Path(tmp).glob("*.tex"):
            names.add(p.name)
            q = tex_dir / p.name
            if not q.exists() or q.read_text() != p.read_text():
                rep.append(dict(file=f"tables/{p.name}", line=0, token="", printed="", kind="table", context="",
                                reason="generated table is stale (differs from a fresh tables.py run)"))
    return names, rep


def run(chapter_files, table_files, ledger, rules, results=RES, check_fresh=True, tex_dir=THESIS / "tables"):
    rep, n_tok, n_q, n_w = [], 0, 0, 0
    gen = set()
    if check_fresh:
        gen, r = fresh_tables(tex_dir)
        rep += r
    texts = {}
    for p in chapter_files:
        texts[p.name] = remove_args(strip_comments(p.read_text()))
    for p in table_files:
        raw = p.read_text()
        is_gen = p.name in gen or raw.lstrip().startswith(GEN_HEADERS)
        t = strip_comments(raw)
        texts[f"tables/{p.name}"] = remove_args(caption_only(t) if is_gen else t)
    ledger = ledger.copy()
    ledger["tex"] = ledger["tex"].fillna("").astype(str).str.strip()
    for f in sorted(set(ledger["tex"]) - set(texts)):
        for _, row in ledger[ledger.tex == f].iterrows():
            rep.append(dict(file=f, line=0, token="", printed="", kind="ledger", context=str(row["id"]),
                            reason="dead row: file not linted (one file per row)"))
    for fname, text in texts.items():
        r, q, w = lint_file(fname, text, ledger[ledger.tex == fname], rules, results)
        rep += r
        n_q += q
        n_w += w
        n_tok += len(tokens(text))
    for i, k in ledger.groupby("id").size().items():
        if k > 1:
            rep.append(dict(file="", line=0, token="", printed="", kind="ledger", context=str(i), reason="duplicate ledger id"))
    return pd.DataFrame(rep, columns=["file", "line", "token", "printed", "kind", "context", "reason"]), n_tok, n_q, n_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fresh", action="store_true")
    a = ap.parse_args()
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=["id", "tex", "anchor", "printed", "note"])
    rules = load_whitelist()
    chapters = sorted((THESIS / "chapters").glob("*.tex"))
    tables = sorted((THESIS / "tables").glob("*.tex"))
    rep, n_tok, n_q, n_w = run(chapters, tables, ledger, rules, check_fresh=not a.no_fresh)
    rep.to_csv(REPORT, index=False)
    print(f"{n_tok} tokens, {n_q} quoted, {n_w} whitelisted, {len(rep)} failures (report: {REPORT.relative_to(HERE)})")
    if len(rep):
        print(rep.groupby("reason").size().to_string())
    sys.exit(1 if len(rep) else 0)


if __name__ == "__main__":
    main()
