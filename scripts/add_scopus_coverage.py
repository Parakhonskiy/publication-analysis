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

SCOPUS_COLS = ["Scopus_Type", "Scopus_Coverage", "Scopus_Edition"]


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

    # Every type this time, not only "journal" — that filter is the defect.
    new_rows = []
    missing = sc[~sc.index.isin(used)]
    for i, r in missing.iterrows():
        issns = r["issns"]
        row = {c: None for c in db.columns}
        row.update({
            "Name": r["Title"],
            "ISSN": issns[0] if issns else "",
            "EISSN": issns[1] if len(issns) > 1 else "",
            "Normalized Journal": r["norm"],
            "Journal_Abbreviation": iso4_fallback(r["Title"]),
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
