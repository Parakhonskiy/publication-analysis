"""
Add Scopus source type and coverage years to journal_database.csv, and bring
in the SCImago sources that are not journals.

Why this exists
  `update_sjr.py` appends only sources whose SCImago type is "journal". The
  2025 export holds 32,193 sources, of which 1,781 are book series,
  conference proceedings and trade journals — all of them indexed by Scopus,
  all of them with their coverage years filled in, and none of them in our
  database. That is why the app's Scopus count came out lower than Scopus's
  own for a pilot participant: two of his papers are in a conference series
  we had simply never imported (publication-web issue #109).

  It also records, for every source it can match, the `Coverage` field —
  the years Scopus actually covers. Publication-web claims coverage only for
  years it has evidence for (its D12), and this is that evidence, from the
  file we already download rather than from a hand-written row.

What it writes
  Scopus_Type      journal | book series | conference and proceedings |
                   trade journal
  Scopus_Coverage  the SCImago string, **verbatim**: "2019-2026", but also
                   "1974, 1977-1985, 1990-1991, 1994-2026". 7,581 of the
                   32,193 sources have gaps like that, so flattening it to a
                   first and last year would claim years Scopus does not
                   cover. The consumer parses the ranges.

Typical use

    python scripts/add_scopus_coverage.py \
        --copy-to-web ../publication-web/data/journal_database.csv

Run it once per SCImago edition. `update_sjr.py` carries the same columns
for every edition after this one, so this script is the migration, not the
routine.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from update_jif import PROCESSED, RAW, REPORTS, clean_issn, iso4_fallback, \
    normalize_journal_name
from update_sjr import SJR_COLS, hyphen_issn, load_scimago

SCOPUS_COLS = ["Scopus_Type", "Scopus_Coverage", "Scopus_Edition",
               "Scopus_Discontinued"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scimago-file", type=Path, default=None,
                    help="default: newest data/raw/scimagojr*.csv")
    ap.add_argument("--db", type=Path, default=PROCESSED / "journal_database.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --db")
    ap.add_argument("--copy-to-web", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    paths = ([args.scimago_file] if args.scimago_file
             else sorted(RAW.glob("scimagojr*.csv")))
    editions = [(int(re.search(r"(20\d\d)", p.name).group(1)), p) for p in paths]
    editions.sort()                      # oldest first; the newest wins
    year = editions[-1][0]
    out_path = args.out or args.db

    db = pd.read_csv(args.db, low_memory=False)
    before = len(db)

    frames = []
    for ed_year, p in editions:
        one = load_scimago(p)
        one["Scopus_Type"] = one["Type"]
        one["Scopus_Coverage"] = one["Coverage"].fillna("").map(
            lambda s: str(s).strip())
        one["Scopus_Edition"] = float(ed_year)
        if ed_year != year:
            # Coverage is a Scopus fact and travels; a score or a quartile
            # belongs to the edition that measured it and does not (I7).
            for c in SJR_COLS:
                one[c] = None
        print(f"SCImago {ed_year}: {len(one)} sources from {p.name}")
        frames.append(one)

    # Newest last, so a source present in several editions keeps the newest
    # row — and one only an older edition knows is still here.
    #
    # The review of PR #111 asked for the *longest* coverage instead, on the
    # ground that a file states coverage as of its download date rather than
    # its edition year. Refused, and measured before refusing: across the
    # three editions exactly **4** sources have a newest-edition coverage
    # that ends earlier than an older edition's, all four by one year
    # (2022 against 2023), and nothing says which of the two statements
    # SCImago corrected. With the coverage years now outranking the SJR
    # score (D15), an extra year is a false claim of the kind the owner has
    # just confirmed in his own export, while a missing one is a claim not
    # made. So the newest edition's statement stands.
    sc = pd.concat(frames, ignore_index=True)
    sc = sc.drop_duplicates("Sourceid", keep="last").reset_index(drop=True)
    print(f"union of {len(editions)} edition(s): {len(sc)} sources")
    print(sc["Type"].value_counts().to_string())
    older = int((sc["Scopus_Edition"] != float(year)).sum())
    print(f"known only from an older edition: {older}")
    print(f"database: {before} rows")

    for c in SCOPUS_COLS:
        if c not in db.columns:
            db[c] = None

    # Same matching as update_sjr: identifier first, then a title that is
    # unique in SCImago. A title that names two sources resolves nothing.
    by_issn: dict[str, int] = {}
    for i, lst in zip(sc.index, sc["issns"]):
        for k in lst:
            by_issn.setdefault(k, i)
    norm_counts = sc["norm"].value_counts()
    by_norm = {n: i for n, i in zip(sc["norm"], sc.index) if norm_counts[n] == 1}

    db["ISSN"] = clean_issn(db["ISSN"])
    db["EISSN"] = clean_issn(db["EISSN"])

    # The marker inside a name, left by an earlier run of update_sjr.py.
    #
    # `load_scimago` strips "(discontinued)" from the title now, because a
    # record whose venue is the plain name has to match the row. These rows
    # were appended before it did, and they are in the committed database:
    # `lookup_journal("AgBioForum")` finds nothing, and so does
    # "American Journal of Health Behavior" (publication-web DATA-18).
    #
    # Stripping changes the normalised key, so a row whose plain name is
    # already in the database becomes a duplicate. The marker row is the
    # poorer of the two — appended from SCImago, no impact factor — so it
    # is the one that goes.
    marked = db["Name"].fillna("").str.contains("(discontinued)", case=False,
                                                regex=False)
    if marked.any():
        db.loc[marked, "Name"] = db.loc[marked, "Name"].str.replace(
            r"\s*\(discontinued\)\s*", " ", case=False, regex=True
        ).str.strip()
        db.loc[marked, "Scopus_Discontinued"] = True
        plain = set(db.loc[~marked, "Name"].fillna("").map(
            normalize_journal_name))
        now_duplicate = marked & db["Name"].map(
            normalize_journal_name).isin(plain)
        print(f"stripping the marker from {int(marked.sum())} names, "
              f"dropping {int(now_duplicate.sum())} that the plain name "
              f"already answers")
        db = db[~now_duplicate].reset_index(drop=True)

    db["Normalized Journal"] = db["Name"].map(normalize_journal_name)

    matched = 0
    used: set[int] = set()
    for idx, (a, b, n) in enumerate(zip(db["ISSN"], db["EISSN"],
                                        db["Normalized Journal"])):
        i = by_issn.get(a) if a else None
        if i is None and b:
            i = by_issn.get(b)
        if i is None:
            i = by_norm.get(n)
        if i is None:
            continue
        used.add(i)
        matched += 1
        r = sc.loc[i]
        for c in SCOPUS_COLS:
            db.at[idx, c] = r[c]

    # The abbreviations an earlier run invented, cleared where they do harm.
    #
    # `update_sjr.py` used to give every Scopus-only source it appended an
    # ISO-4 guess at its abbreviation, and those rows are in the committed
    # database. The guess collides with the row the same journal already has
    # under its Web of Science name — and sometimes with a different journal
    # altogether: `algal res.` names both Algal Research (JIF 5.0) and an
    # appended duplicate, and `appl. econ.` names APPLIED ECONOMICS (JIF
    # 2.6) and Applied Econometrics. An ambiguous abbreviation yields no
    # metrics at all (Issue #10, I5), so the guess costs a journal the
    # impact factor that is really its own.
    #
    # Cleared where both hold: the row has no impact factor, so no JCR
    # edition ever named it, and its abbreviation is exactly the guess its
    # own title produces, so nothing was looked up to get it. 9,743 rows
    # answer to that, and 70 ambiguous abbreviations go with them. The
    # sources stay findable by name, which is how a reader meets them.
    guess = db["Name"].fillna("").map(iso4_fallback).fillna("")
    have   = db["Journal_Abbreviation"].fillna("")
    no_jif = db["JIF"].isna() | (db["JIF"].astype(str).str.strip() == "")
    invented = no_jif & (have != "") & (
        have.str.strip().str.lower() == guess.str.strip().str.lower())
    print(f"clearing {int(invented.sum())} invented abbreviations "
          f"(no impact factor, abbreviation is the ISO-4 guess)")
    db.loc[invented, "Journal_Abbreviation"] = None

    # Every type this time, not only "journal" — that filter is the defect.
    new_rows = []
    missing = sc[~sc.index.isin(used)]
    # The other half of DATA-4, and this one is real. A title can name two
    # sources under two `Sourceid`s — a reissue, a change of publisher — and
    # the duplicate drop below keeps the *first* row, which in edition order
    # is the oldest file's. 17 titles are in that position and in every one
    # the newest edition covers further, so the append was handing out the
    # shorter coverage. Newest first here, so `keep="first"` keeps it.
    missing = missing.sort_values("Scopus_Edition", ascending=False,
                                  kind="stable")
    for i, r in missing.iterrows():
        issns = r["issns"]
        row = {c: None for c in db.columns}
        row.update({
            "Name": r["Title"],
            "ISSN": issns[0] if issns else "",
            "EISSN": issns[1] if len(issns) > 1 else "",
            "Normalized Journal": r["norm"],
            # No `Journal_Abbreviation`: see update_sjr.py. The ISO-4 guess
            # collides with the row the same journal already has under its
            # Web of Science name, and an ambiguous abbreviation yields no
            # metrics at all (DATA-5, DATA-6, Issue #10).
            "SJR_Year": float(year),
            "Scopus_Type": r["Scopus_Type"],
            "Scopus_Coverage": r["Scopus_Coverage"],
            "Scopus_Edition": r["Scopus_Edition"],
        })
        for c in SJR_COLS:
            row[c] = r[c]
        new_rows.append(row)

    by_type = missing["Type"].value_counts().to_dict()
    print(f"\nmatched existing rows: {matched}")
    print(f"appending {len(new_rows)} sources not in the database: {by_type}")
    with_cov = int((db["Scopus_Coverage"].fillna("") != "").sum())
    print(f"rows carrying coverage years after the match: {with_cov}")

    if new_rows:
        db = pd.concat([db, pd.DataFrame(new_rows)], ignore_index=True)
        db = db.drop_duplicates("Normalized Journal", keep="first")

    gaps = int(db["Scopus_Coverage"].fillna("").str.contains(",").sum())
    print(f"rows whose coverage has gaps (several ranges): {gaps}")
    print(f"database: {before} -> {len(db)} rows")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_csv(out_path, index=False)
    print(f"written {out_path}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"scopus_coverage_{year}.md"
    report.write_text(
        f"# Scopus type and coverage, SCImago {year}\n\n"
        f"Editions read: {', '.join(p.name for _y, p in editions)}\n\n"
        f"Union: {len(sc)} sources, {older} of them known only from an "
        f"older edition — coverage travels, a score or a quartile does "
        f"not (I7).\n\n"
        f"| | |\n|---|---|\n"
        f"| database rows before | {before} |\n"
        f"| matched and annotated | {matched} |\n"
        f"| appended (all types) | {len(new_rows)} |\n"
        f"| of those, by type | {by_type} |\n"
        f"| rows with coverage years | {with_cov} |\n"
        f"| rows whose coverage has gaps | {gaps} |\n"
        f"| database rows after | {len(db)} |\n\n"
        "`Scopus_Coverage` is the SCImago string verbatim, gaps included: "
        "flattening it to a first and last year would claim years Scopus "
        "does not cover.\n", encoding="utf-8")
    print(f"written {report}")

    if args.copy_to_web:
        import shutil
        shutil.copyfile(out_path, args.copy_to_web)
        print(f"copied to {args.copy_to_web}")


if __name__ == "__main__":
    main()
