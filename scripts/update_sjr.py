"""
Refresh the SCImago (Scopus) part of journal_database.csv from a newer
SCImago Journal Rank export, keeping the previous values as history columns.

Typical use (SCImago 2025 edition):

    python scripts/update_sjr.py --scimago-file "data/raw/scimagojr 2025.csv" \
        --copy-to-web ../publication-web/data/journal_database.csv

What it does
  1. Reads the SCImago CSV (semicolon-separated; "Issn" holds one or two
     8-digit ISSNs; SJR uses a decimal comma).
  2. Matches every DB row by ISSN / eISSN, then by normalised title
     (only when the title is unique in SCImago).
  3. For matched rows overwrites  SJR_Score, SJR_Quartile, H_Index,
     SJR_Categories, SJR_Areas  and sets  SJR_Year; the previous score and
     quartile are kept in  SJR_Score_<old year> / SJR_Quartile_<old year>.
     Unmatched rows keep their old values (SJR_Year stays old).
  4. Appends SCImago sources of type "journal" that are not in the DB yet
     (Scopus-only journals — no JIF, but the web app can then flag them as
     Scopus-indexed).  Skip with --no-append.
  5. Writes the DB and a change report to reports/sjr_update_<year>.md.

The SCImago year is taken from --sjr-year or parsed from the file name
("scimagojr 2025.csv" → 2025).  The previous year is read from the DB's
SJR_Year column, or --prev-year if the column does not exist yet.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from update_jif import normalize_journal_name, iso4_fallback, clean_issn, PROCESSED, RAW, REPORTS

SJR_COLS = ["SJR_Score", "SJR_Quartile", "H_Index", "SJR_Categories", "SJR_Areas"]


def _num(v) -> float | None:
    s = str(v).strip().replace(",", ".")
    if s in ("", "nan", "None", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def hyphen_issn(raw: str) -> str:
    s = raw.strip().upper()
    return f"{s[:4]}-{s[4:]}" if len(s) == 8 and "-" not in s else s


def load_scimago(path: Path) -> pd.DataFrame:
    sc = pd.read_csv(path, sep=";", dtype=str)
    sc = sc.rename(columns={"SJR Best Quartile": "SJR_Quartile", "H index": "H_Index",
                            "Categories": "SJR_Categories", "Areas": "SJR_Areas"})
    sc["SJR_Score"] = sc["SJR"].map(_num)
    sc["H_Index"]   = sc["H_Index"].map(_num)
    sc["SJR_Quartile"] = sc["SJR_Quartile"].where(sc["SJR_Quartile"].isin(["Q1", "Q2", "Q3", "Q4"]), None)
    sc["issns"] = sc["Issn"].fillna("").map(lambda s: [hyphen_issn(x) for x in s.split(",") if x.strip()])
    sc["norm"]  = sc["Title"].map(normalize_journal_name)
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scimago-file", type=Path, default=None, help="default: newest data/raw/scimagojr*.csv")
    ap.add_argument("--sjr-year", type=int, default=None, help="edition year; default parsed from file name")
    ap.add_argument("--prev-year", type=int, default=2023, help="year of the SJR data currently in the DB (if no SJR_Year column)")
    ap.add_argument("--db", type=Path, default=PROCESSED / "journal_database.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --db")
    ap.add_argument("--copy-to-web", type=Path, default=None)
    ap.add_argument("--no-append", action="store_true", help="do not add Scopus-only journals")
    args = ap.parse_args()

    sc_path = args.scimago_file or sorted(RAW.glob("scimagojr*.csv"))[-1]
    year = args.sjr_year or int(re.search(r"(20\d\d)", sc_path.name).group(1))
    out_path = args.out or args.db

    db = pd.read_csv(args.db, low_memory=False)
    if "SJR_Year" in db.columns and pd.to_numeric(db["SJR_Year"], errors="coerce").max() >= year:
        raise SystemExit(f"DB already carries SJR {year} — nothing to do.")
    prev = int(pd.to_numeric(db["SJR_Year"], errors="coerce").max()) if "SJR_Year" in db.columns else args.prev_year
    if "SJR_Year" not in db.columns:
        db["SJR_Year"] = db["SJR_Score"].notna().map(lambda has: float(prev) if has else None)
    for c in SJR_COLS:
        if c not in db.columns:
            db[c] = None
    db[f"SJR_Score_{prev}"]    = db["SJR_Score"]
    db[f"SJR_Quartile_{prev}"] = db["SJR_Quartile"]

    sc = load_scimago(sc_path)
    print(f"SCImago {year}: {len(sc)} sources ({(sc['Type'] == 'journal').sum()} journals) from {sc_path.name}")

    by_issn: dict[str, int] = {}
    for i, lst in zip(sc.index, sc["issns"]):
        for k in lst:
            by_issn.setdefault(k, i)
    norm_counts = sc["norm"].value_counts()
    by_norm = {n: i for n, i in zip(sc["norm"], sc.index) if norm_counts[n] == 1}

    db["ISSN"]  = clean_issn(db["ISSN"])
    db["EISSN"] = clean_issn(db["EISSN"])
    db["Normalized Journal"] = db["Name"].map(normalize_journal_name)

    m_issn = m_name = 0
    used: set[int] = set()
    for idx, (a, b, n) in enumerate(zip(db["ISSN"], db["EISSN"], db["Normalized Journal"])):
        i = by_issn.get(a) if a else None
        if i is None and b:
            i = by_issn.get(b)
        if i is not None:
            m_issn += 1
        else:
            i = by_norm.get(n)
            if i is not None:
                m_name += 1
        if i is None:
            continue
        used.add(i)
        r = sc.loc[i]
        for c in SJR_COLS:
            db.at[idx, c] = r[c]
        db.at[idx, "SJR_Year"] = float(year)
        if not a and r["issns"]:
            db.at[idx, "ISSN"] = r["issns"][0]

    # Scopus-only journals → append
    new_rows = []
    if not args.no_append:
        for i, r in sc[(sc["Type"] == "journal") & (~sc.index.isin(used))].iterrows():
            row = {c: None for c in db.columns}
            issns = r["issns"]
            row.update({
                "Name": r["Title"], "ISSN": issns[0] if issns else "", "EISSN": issns[1] if len(issns) > 1 else "",
                "Normalized Journal": r["norm"], "Journal_Abbreviation": iso4_fallback(r["Title"]),
                "SJR_Year": float(year),
            })
            for c in SJR_COLS:
                row[c] = r[c]
            new_rows.append(row)
        if new_rows:
            db = pd.concat([db, pd.DataFrame(new_rows)], ignore_index=True)
            db = db.drop_duplicates("Normalized Journal", keep="first")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_csv(out_path, index=False)
    print(f"Written {out_path} — {len(db)} rows")
    if args.copy_to_web:
        shutil.copy2(out_path, args.copy_to_web)
        print(f"Copied to {args.copy_to_web}")

    upd = db[db["SJR_Year"] == year]
    both = upd[upd[f"SJR_Score_{prev}"].notna() & upd["SJR_Score"].notna()].copy()
    both["delta"] = both["SJR_Score"] - both[f"SJR_Score_{prev}"]
    q_changed = (upd[f"SJR_Quartile_{prev}"].notna() & (upd[f"SJR_Quartile_{prev}"] != upd["SJR_Quartile"])).sum()
    kept_old = db[(db["SJR_Year"] == prev)]
    lines = [
        f"# SJR update: SCImago {prev} → {year}", "",
        f"- Source: {sc_path.name} — {len(sc)} sources",
        f"- DB rows matched by ISSN/eISSN: {m_issn}; by unique title: {m_name}",
        f"- Scopus-only journals appended: {len(new_rows)}",
        f"- DB rows keeping SJR {prev} (not in new file): {len(kept_old)}",
        f"- SJR quartile changed: {int(q_changed)}",
        f"- Median SJR change: {both['delta'].median():+.3f} (mean {both['delta'].mean():+.3f}); "
        f"{int((both['delta'] > 0).sum())} up, {int((both['delta'] < 0).sum())} down",
        "", "## Largest SJR increases", "",
        both.nlargest(15, "delta")[["Name", f"SJR_Score_{prev}", "SJR_Score", "delta"]].to_string(index=False),
        "", "## Largest SJR decreases", "",
        both.nsmallest(15, "delta")[["Name", f"SJR_Score_{prev}", "SJR_Score", "delta"]].to_string(index=False),
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    rep = REPORTS / f"sjr_update_{year}.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nFull report → {rep}")


if __name__ == "__main__":
    main()
