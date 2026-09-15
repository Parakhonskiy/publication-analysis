"""
Update journal_database.csv with a newer JCR impact-factor release, keeping
the previous edition's values as historical columns.

Typical use (JCR 2026 release = JIF for citation year 2025):

    python scripts/update_jif.py \
        --jcr-file "data/raw/DOC-20260618-WA0000.xlsx" --jif-year 2025 \
        --copy-to-web ../publication-web/data/journal_database.csv

What it does
  1. Archives the current DB as  data/processed/journal_database_<old JCR year>.csv
     (only if that archive does not exist yet).
  2. Matches every journal of the new release to the DB by ISSN / eISSN,
     then by normalised name.
  3. For matched journals:  JIF, Area (JIF quartile), Total Cites, JCR Year
     are overwritten with the new values; the previous JIF / quartile are
     kept in  JIF_<old year>  /  Quartile_<old year>.
     Unmatched DB journals keep their old values (JCR Year stays old).
  4. Journals that exist only in the new release are appended, with ISO 4
     abbreviation from ABB.csv (or generated) and SCImago data by ISSN.
  5. Recomputes 'Normalized Journal' for every row with the web app's
     normaliser and collapses rows that turn out to be duplicates.
  6. Writes the updated DB and a change report to reports/.

Columns JIF5Years, JCI, JIF_NoSelfCite, Rank are NOT present in the new
source and therefore keep the previous edition's values.

Input format of --jcr-file: the "JCR <year>" Excel table with a title row
followed by a header row: journal name, ISSN, eISSN, WoS category, index
(SCIE/SSCI/ESCI/AHCI), total cites, JIF, quartile — one row per
journal × category (quartile differs per category; the best one is kept).
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW       = ROOT / "data" / "raw"
REPORTS   = ROOT / "reports"


# ── normalisation — verbatim copy of publication-web/core/journal_lookup.py
#    normalise_journal().  The web app looks journals up by this key, so the
#    stored 'Normalized Journal' column MUST be produced by the same function.
def normalize_journal_name(name) -> str:
    if not name or not isinstance(name, str):
        return ""
    # Strip trailing Scopus location tags: "(Bristol)", "(United States)", …
    name = re.sub(r"\s*\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\)\s*$", "", name)
    name = name.replace("\&", " and ")
    name = name.replace("&", " and ")
    name = re.sub(r"-{1,3}|–|—", " ", name)
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name.startswith("the "):
        name = name[4:]
    return name


def iso4_fallback(journal_name: str) -> str:
    """Approximate ISO 4 abbreviation for journals missing from ABB.csv
    (same heuristic as notebook 01, section 5d)."""
    abbr_map = {
        "journal": "J.", "journals": "J.", "international": "Int.",
        "review": "Rev.", "reviews": "Rev.", "annals": "Ann.",
        "bulletin": "Bull.", "medicine": "Med.", "medical": "Med.",
        "clinical": "Clin.", "american": "Am.", "european": "Eur.",
        "science": "Sci.", "sciences": "Sci.", "technology": "Technol.",
        "engineering": "Eng.", "communications": "Commun.",
        "proceedings": "Proc.", "transactions": "Trans.",
        "letters": "Lett.", "advances": "Adv.", "reports": "Rep.",
        "report": "Rep.", "research": "Res.", "studies": "Stud.",
        "materials": "Mater.", "chemistry": "Chem.", "chemical": "Chem.",
        "physics": "Phys.", "physical": "Phys.", "biology": "Biol.",
        "biological": "Biol.", "environmental": "Environ.", "energy": "Energy",
        "applied": "Appl.", "nature": "Nat.", "nanotechnology": "Nanotechnol.",
    }
    stop = {"and", "of", "for", "the", "on", "in", "a", "an", "&"}
    words = re.split(r"\s+", journal_name.strip())
    if len(words) == 1:
        w = words[0]
        return w.capitalize() if w.isupper() else w
    out = []
    for w in words:
        lw = w.lower().strip(",.:;")
        if lw in stop:
            continue
        if lw in abbr_map:
            out.append(abbr_map[lw])
        elif lw.endswith("ology"):
            out.append(lw[:-4].capitalize() + "ol.")
        elif len(lw) > 6:
            out.append(lw[:4].capitalize() + ".")
        else:
            out.append(w if w.isupper() and len(w) <= 4 else w.capitalize())
    return " ".join(out) if out else journal_name


def clean_issn(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    return s.where(~s.isin(["N/A", "NAN", "NONE", ""]), "")


def parse_jif(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("<"):            # "<0.1"
            return 0.05
        try:
            return float(v)
        except ValueError:
            return None
    return float(v)


# ── load the new JCR release ─────────────────────────────────────────────────
def load_jcr_release(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=1)
    raw.columns = ["Name", "ISSN", "EISSN", "WoS_Category", "Index",
                   "Total_Cites", "JIF", "Quartile"][: len(raw.columns)]
    raw["Name"]  = raw["Name"].astype(str).str.strip()
    raw["ISSN"]  = clean_issn(raw["ISSN"])
    raw["EISSN"] = clean_issn(raw["EISSN"])
    raw["JIF"]   = raw["JIF"].map(parse_jif)
    raw["Qnum"]  = raw["Quartile"].astype(str).str.extract(r"Q([1-4])")[0].astype(float)

    # one row per journal; best (lowest) quartile across categories
    g = raw.groupby("Name", sort=False)
    out = g.agg(
        ISSN=("ISSN", "first"), EISSN=("EISSN", "first"),
        Index=("Index", "first"), Total_Cites=("Total_Cites", "first"),
        JIF=("JIF", "first"), Qnum=("Qnum", "min"),
        WoS_Categories=("WoS_Category", lambda s: "; ".join(dict.fromkeys(s.dropna().astype(str)))),
    ).reset_index()
    out["Quartile"] = out["Qnum"].map(lambda q: f"Q{int(q)}" if pd.notna(q) else None)
    out["norm"] = out["Name"].map(normalize_journal_name)
    return out.drop(columns="Qnum")


# ── SCImago (for journals new to the DB) ─────────────────────────────────────
def load_scimago() -> pd.DataFrame | None:
    files = sorted(RAW.glob("scimagojr*.csv"))
    if not files:
        return None
    sc = pd.read_csv(files[-1], sep=";", dtype=str)
    rows = []
    for _, r in sc.iterrows():
        for issn in str(r.get("Issn", "")).split(","):
            issn = issn.strip().upper()
            if len(issn) == 8:
                issn = issn[:4] + "-" + issn[4:]
            if not issn:
                continue
            rows.append({
                "issn": issn,
                "SJR_Score": float(str(r.get("SJR", "")).replace(",", ".")) if str(r.get("SJR", "")).strip() not in ("", "nan") else None,
                "SJR_Quartile": r.get("SJR Best Quartile"),
                "H_Index": float(r["H index"]) if str(r.get("H index", "")).strip() not in ("", "nan") else None,
                "SJR_Categories": r.get("Categories"),
                "SJR_Areas": r.get("Areas"),
            })
    return pd.DataFrame(rows).drop_duplicates("issn").set_index("issn")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jcr-file", required=True, type=Path, help="new JCR release (.xlsx)")
    ap.add_argument("--jif-year", required=True, type=int, help="citation year of the new JIF, e.g. 2025 for the JCR 2026 release")
    ap.add_argument("--db", type=Path, default=PROCESSED / "journal_database.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --db")
    ap.add_argument("--copy-to-web", type=Path, default=None, help="also copy the result here (publication-web/data/journal_database.csv)")
    args = ap.parse_args()
    out_path = args.out or args.db

    db = pd.read_csv(args.db, low_memory=False)
    old_year = int(pd.to_numeric(db["JCR Year"], errors="coerce").dropna().mode().iloc[0])
    if old_year >= args.jif_year:
        raise SystemExit(f"DB already at JCR Year {old_year} — nothing to do for {args.jif_year}.")

    # 1. archive previous edition
    archive = PROCESSED / f"journal_database_{old_year}.csv"
    if not archive.exists():
        shutil.copy2(args.db, archive)
        print(f"Archived previous edition → {archive.name}")
    else:
        print(f"Archive already exists, left untouched: {archive.name}")

    # 2. load and match
    new = load_jcr_release(args.jcr_file)
    print(f"New release: {len(new)} journals ({new['JIF'].notna().sum()} with JIF)")

    db["ISSN"]  = clean_issn(db["ISSN"])
    db["EISSN"] = clean_issn(db["EISSN"])
    # Recompute with the web app's normaliser (the notebook's older, looser
    # version left ~700 names unmatched on the site).
    db["Normalized Journal"] = db["Name"].map(normalize_journal_name)

    by_issn: dict[str, int] = {}
    for idx, (a, b) in enumerate(zip(db["ISSN"], db["EISSN"])):
        for k in (a, b):
            if k and k not in by_issn:
                by_issn[k] = idx
    by_norm: dict[str, int] = {}
    for idx, n in enumerate(db["Normalized Journal"]):
        if n and n not in by_norm:
            by_norm[n] = idx

    hist_jif, hist_q = f"JIF_{old_year}", f"Quartile_{old_year}"
    db[hist_jif] = db["JIF"]
    db[hist_q]   = db["Area"]
    if "WoS_Index" not in db.columns:
        db["WoS_Index"] = None
    if "WoS_Categories" not in db.columns:
        db["WoS_Categories"] = None

    matched_issn = matched_name = 0
    updates: dict[int, pd.Series] = {}
    unmatched: list[pd.Series] = []
    for _, r in new.iterrows():
        idx = by_issn.get(r["ISSN"]) if r["ISSN"] else None
        if idx is None and r["EISSN"]:
            idx = by_issn.get(r["EISSN"])
        if idx is not None:
            matched_issn += 1
        else:
            idx = by_norm.get(r["norm"])
            if idx is not None:
                matched_name += 1
        if idx is None:
            unmatched.append(r)
        elif idx not in updates:          # first new-release row wins
            updates[idx] = r

    # 3. apply updates
    for idx, r in updates.items():
        db.at[idx, "JIF"]            = r["JIF"]
        db.at[idx, "JCR Year"]       = float(args.jif_year)
        db.at[idx, "Total Cites"]    = r["Total_Cites"]
        db.at[idx, "WoS_Index"]      = r["Index"]
        db.at[idx, "WoS_Categories"] = r["WoS_Categories"]
        if r["Quartile"]:
            db.at[idx, "Area"] = r["Quartile"]
        if not db.at[idx, "ISSN"] and r["ISSN"]:
            db.at[idx, "ISSN"] = r["ISSN"]
        if not db.at[idx, "EISSN"] and r["EISSN"]:
            db.at[idx, "EISSN"] = r["EISSN"]

    # 4. append journals that are new to the DB
    abb = pd.read_csv(RAW / "ABB.csv")
    abb_map = {normalize_journal_name(n): a for n, a in zip(abb["Journal Name"], abb["ISO 4 abbreviation"])}
    scimago = load_scimago()
    new_rows = []
    for r in unmatched:
        row = {c: None for c in db.columns}
        row.update({
            "Name": r["Name"], "JCR Year": float(args.jif_year),
            "ISSN": r["ISSN"], "EISSN": r["EISSN"],
            "Total Cites": r["Total_Cites"], "JIF": r["JIF"], "Area": r["Quartile"],
            "Normalized Journal": r["norm"],
            "Journal_Abbreviation": abb_map.get(r["norm"]) or iso4_fallback(r["Name"]),
            "WoS_Index": r["Index"], "WoS_Categories": r["WoS_Categories"],
        })
        if scimago is not None:
            for k in (r["ISSN"], r["EISSN"]):
                if k and k in scimago.index:
                    for c in ("SJR_Score", "SJR_Quartile", "H_Index", "SJR_Categories", "SJR_Areas"):
                        row[c] = scimago.at[k, c]
                    break
        new_rows.append(row)
    if new_rows:
        db = pd.concat([db, pd.DataFrame(new_rows)], ignore_index=True)

    # 5. collapse duplicate keys — the notebook appended "ABB-only" placeholder
    #    rows (same journal, comma-spelled, no JIF) that the stricter normaliser
    #    now reveals as twins of real rows.  Keep the richest row per key and
    #    inherit an abbreviation from the dropped twin if needed.
    db["_score"] = db["JIF"].notna().astype(int) * 4 + db["SJR_Score"].notna().astype(int) * 2                  + db["Journal_Abbreviation"].notna().astype(int)
    db = db.sort_values(["Normalized Journal", "_score"], ascending=[True, False], kind="stable")
    abbr_by_key = db.dropna(subset=["Journal_Abbreviation"]).drop_duplicates("Normalized Journal")                     .set_index("Normalized Journal")["Journal_Abbreviation"]
    n_before = len(db)
    db = db.drop_duplicates("Normalized Journal", keep="first").copy()
    miss = db["Journal_Abbreviation"].isna()
    db.loc[miss, "Journal_Abbreviation"] = db.loc[miss, "Normalized Journal"].map(abbr_by_key)
    db = db.drop(columns="_score").sort_index()
    dropped_dups = n_before - len(db)
    print(f"Collapsed {dropped_dups} duplicate rows")

    # 6. write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_csv(out_path, index=False)
    print(f"Written {out_path} — {len(db)} rows")

    if args.copy_to_web:
        shutil.copy2(out_path, args.copy_to_web)
        print(f"Copied to {args.copy_to_web}")

    # report
    upd = db.loc[db.index.intersection(list(updates.keys()))]
    changed = upd[(upd[hist_jif].notna()) & (upd["JIF"].notna())].copy()
    changed["delta"] = changed["JIF"] - changed[hist_jif]
    changed["pct"]   = changed["delta"] / changed[hist_jif].replace(0, float("nan")) * 100
    q_changed = (upd[hist_q].fillna("") != upd["Area"].fillna("")) & upd[hist_q].notna()
    kept_old = db[(db["JCR Year"] == old_year) & db[hist_jif].notna()]

    lines = [
        f"# JIF update: JCR {old_year} → JIF {args.jif_year}",
        "",
        f"- New release: {len(new)} journals",
        f"- Matched by ISSN/eISSN: {matched_issn}; by normalised name: {matched_name}",
        f"- Appended (new to DB): {len(new_rows)}; duplicate rows collapsed: {dropped_dups}",
        f"- DB journals without a {args.jif_year} value (kept {old_year} JIF, JCR Year={old_year}): {len(kept_old)}",
        f"- Quartile changed: {int(q_changed.sum())}",
        f"- Median JIF change among matched: {changed['delta'].median():+.2f} "
        f"(mean {changed['delta'].mean():+.2f}); {int((changed['delta'] > 0).sum())} up, {int((changed['delta'] < 0).sum())} down",
        "",
        "## Largest increases", "",
        changed.nlargest(15, "delta")[["Name", hist_jif, "JIF", "delta"]].to_string(index=False),
        "", "## Largest decreases", "",
        changed.nsmallest(15, "delta")[["Name", hist_jif, "JIF", "delta"]].to_string(index=False),
        "", "## New journals with highest JIF", "",
        pd.DataFrame(new_rows).nlargest(15, "JIF")[["Name", "WoS_Index", "JIF", "Area"]].to_string(index=False) if new_rows else "(none)",
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    rep = REPORTS / f"jif_update_{args.jif_year}.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\nFull report → {rep}")


if __name__ == "__main__":
    main()
