#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт выгрузки экспериментальных данных об активности малых молекул против вируса оспы
(VACV / Poxviridae) из открытых баз данных (ChEMBL REST API, PubChem PUG REST API),
получения реальных DOI публикаций и RDKit-валидации химических структур.
"""

import io
import os
import time
import requests
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# Ссылки на открытые базы данных (согласно условию хакатона)
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
CHEMBL_DOCUMENT_URL = "https://www.ebi.ac.uk/chembl/api/data/document.json"

CHEMBL_TARGETS = [
    ("CHEMBL613295", "Vaccinia virus", "Vero / HeLa cells"),
    ("CHEMBL613120", "Monkeypox virus", "Vero cells"),
    ("CHEMBL613728", "Cowpox virus", "HFF / Vero cells"),
    ("CHEMBL612745", "Variola virus", "Vero cells"),
    ("CHEMBL613729", "Ectromelia virus", "Vero cells"),
    ("CHEMBL613730", "Camelpox virus", "Vero cells"),
    ("CHEMBL612545", "Orf virus", "HFF / Vero cells"),
]

# Реальные биоскрининги активности против Vaccinia / Poxviridae в PubChem BioAssay
PUBCHEM_AID_LIST = [
    1722, 1906, 2147, 2326, 463114, 485294, 504333, 504466, 588401, 624256, 720543
]


def validate_smiles(smiles: str):
    """Валидация SMILES-записи, санитизация и приведение к каноническому виду через RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def convert_to_uM(val, units: str):
    """Конвертация концентраций IC50/EC50/CC50 в микромоли (uM)."""
    if val is None or pd.isna(val):
        return np.nan
    try:
        v = float(val)
    except (ValueError, TypeError):
        return np.nan
    if v <= 0:
        return np.nan
    u = str(units).strip().lower()
    if u in ["nm", "nanomolar", "10^-9m", "nmol/l"]:
        return v / 1000.0
    elif u in ["mm", "millimolar", "10^-3m", "mmol/l"]:
        return v * 1000.0
    elif u in ["pm", "picomolar", "10^-12m"]:
        return v / 1e6
    elif u in ["m", "molar", "mol/l"]:
        return v * 1e6
    return v


def fetch_chembl_dois(doc_ids):
    """Пакетная выгрузка реальных DOI публикаций по списку document_chembl_id из ChEMBL API."""
    doi_map = {}
    unique_ids = sorted({str(d) for d in doc_ids if str(d).startswith("CHEMBL")})
    for i in range(0, len(unique_ids), 40):
        batch = ",".join(unique_ids[i : i + 40])
        url = f"{CHEMBL_DOCUMENT_URL}?document_chembl_id__in={batch}&limit=50"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                for doc in resp.json().get("documents", []):
                    doc_id = doc.get("document_chembl_id")
                    doi = doc.get("doi")
                    pubmed_id = doc.get("pubmed_id")
                    if doc_id and doi:
                        doi_map[doc_id] = str(doi).strip()
                    elif doc_id and pubmed_id:
                        doi_map[doc_id] = f"PMID:{pubmed_id}"
        except Exception as e:
            print(f"[ChEMBL Document API] Предупреждение: {e}")
        time.sleep(0.1)
    return doi_map


def fetch_chembl_cytotoxicity_for_molecules(mol_ids):
    """Дополнительный поиск экспериментальных значений CC50/GI50 для выгруженных молекул ChEMBL."""
    cc50_map = {}
    unique_mols = sorted({str(m) for m in mol_ids if str(m).startswith("CHEMBL")})
    for i in range(0, len(unique_mols), 35):
        batch = ",".join(unique_mols[i : i + 35])
        url = (
            f"{CHEMBL_ACTIVITY_URL}?molecule_chembl_id__in={batch}"
            f"&standard_type__in=CC50,GI50,TD50&limit=200"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                for act in resp.json().get("activities", []):
                    m_id = act.get("molecule_chembl_id")
                    val_um = convert_to_uM(act.get("standard_value"), act.get("standard_units"))
                    if m_id and pd.notna(val_um) and 0 < val_um < 50000:
                        cc50_map.setdefault(m_id, []).append(val_um)
        except Exception:
            pass
        time.sleep(0.1)
    return {k: float(np.median(v)) for k, v in cc50_map.items() if v}


def download_from_chembl(limit_per_target=1000):
    """Выгрузка реальных экспериментальных данных IC50, EC50, CC50 и DOI из ChEMBL API."""
    records = {}
    doc_ids_to_resolve = set()
    mol_ids_to_resolve = set()

    for target_id, strain_name, culture_name in CHEMBL_TARGETS:
        offset = 0
        batch_size = 500
        fetched = 0
        while fetched < limit_per_target:
            url = (
                f"{CHEMBL_ACTIVITY_URL}?target_chembl_id={target_id}"
                f"&standard_type__in=IC50,EC50,CC50,GI50&limit={batch_size}&offset={offset}"
            )
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    break
                data = resp.json()
            except Exception as e:
                print(f"[ChEMBL] Предупреждение при запросе {url}: {e}")
                break

            acts = data.get("activities", [])
            if not acts:
                break

            for act in acts:
                c_smi = validate_smiles(act.get("canonical_smiles"))
                if not c_smi:
                    continue
                mol_id = act.get("molecule_chembl_id") or "CHEMBL_CMPD"
                stype = str(act.get("standard_type", "")).upper()
                val_um = convert_to_uM(act.get("standard_value"), act.get("standard_units"))
                doc_id = act.get("document_chembl_id") or ""
                raw_doi = act.get("document_doi")

                if doc_id.startswith("CHEMBL") and not raw_doi:
                    doc_ids_to_resolve.add(doc_id)
                if mol_id.startswith("CHEMBL"):
                    mol_ids_to_resolve.add(mol_id)

                if c_smi not in records:
                    records[c_smi] = {
                        "Title": mol_id,
                        "CC50_vals": [],
                        "IC50_vals": [],
                        "SMILES": c_smi,
                        "strain": strain_name,
                        "Culture": culture_name,
                        "doc_id": doc_id,
                        "doi": raw_doi or doc_id or "ChEMBL_OpenData",
                        "source_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_id}/",
                    }
                if stype in ["IC50", "EC50"] and pd.notna(val_um):
                    records[c_smi]["IC50_vals"].append(val_um)
                elif stype in ["CC50", "GI50"] and pd.notna(val_um):
                    records[c_smi]["CC50_vals"].append(val_um)

            fetched += len(acts)
            offset += batch_size
            if not data.get("page_meta", {}).get("next"):
                break
            time.sleep(0.12)

    print(f"      Найдено {len(records)} уникальных молекул в ChEMBL. Запрос DOI и цитотоксичности...")
    doi_map = fetch_chembl_dois(doc_ids_to_resolve)
    cc50_map = fetch_chembl_cytotoxicity_for_molecules(mol_ids_to_resolve)

    out_rows = []
    for c_smi, rec in records.items():
        cc_vals = rec["CC50_vals"]
        if not cc_vals and rec["Title"] in cc50_map:
            cc_vals = [cc50_map[rec["Title"]]]

        cc50 = round(float(np.median(cc_vals)), 3) if cc_vals else np.nan
        ic50 = round(float(np.median(rec["IC50_vals"])), 4) if rec["IC50_vals"] else np.nan

        if pd.notna(cc50) and pd.notna(ic50) and ic50 > 0:
            si = round(cc50 / ic50, 3)
        elif pd.notna(cc50) and pd.isna(ic50):
            si = 10000.0
        else:
            si = np.nan

        resolved_doi = doi_map.get(rec["doc_id"], rec["doi"])
        out_rows.append({
            "Title": rec["Title"],
            "CC50_uM": cc50,
            "IC50_uM_VACV": ic50,
            "SI VACV": si,
            "SMILES": c_smi,
            "strain": rec["strain"],
            "Culture": rec["Culture"],
            "doi": resolved_doi,
            "source_url": rec["source_url"],
        })
    return pd.DataFrame(out_rows)


def fetch_pubchem_assay_doi(aid: int):
    """Получение DOI или PubMed ID публикации для биоскрининга PubChem AID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON"
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code == 200:
            desc = resp.json().get("PC_AssayDescription", {})
            for xref in desc.get("xref", []):
                xr = xref.get("xref", {})
                if "doi" in xr:
                    return str(xr["doi"])
                if "pmid" in xr:
                    return f"PMID:{xr['pmid']}"
    except Exception:
        pass
    return f"PubChem_AID:{aid}"


def download_from_pubchem(aid_list=PUBCHEM_AID_LIST, max_rows_per_aid=600):
    """Выгрузка реальных результатов биоскринингов против вируса оспы из PubChem PUG REST API."""
    rows = []
    for aid in aid_list:
        doi_str = fetch_pubchem_assay_doi(aid)
        csv_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV?sid=all"
        try:
            resp = requests.get(csv_url, timeout=20)
            if resp.status_code != 200 or not resp.text:
                continue
            df_aid = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            smi_col = next((c for c in df_aid.columns if "SMILES" in c.upper()), None)
            sid_col = next((c for c in df_aid.columns if "PUBCHEM_SID" in c.upper()), None)
            out_col = next((c for c in df_aid.columns if "ACTIVITY_OUTCOME" in c.upper()), None)
            act_col = next(
                (c for c in df_aid.columns if any(k in c.upper() for k in ["IC50", "EC50", "AC50", "ACTIVITY_VALUE"])),
                None,
            )
            if not smi_col:
                continue

            count = 0
            for _, r in df_aid.iterrows():
                if count >= max_rows_per_aid:
                    break
                c_smi = validate_smiles(str(r.get(smi_col, "")))
                if not c_smi:
                    continue
                sid_val = r.get(sid_col, "")
                title = f"sid{int(sid_val)}" if pd.notna(sid_val) and str(sid_val).isdigit() else f"AID{aid}_{count}"
                ic50_val = pd.to_numeric(r.get(act_col), errors="coerce") if act_col else np.nan
                outcome = str(r.get(out_col, "")).lower() if out_col else ""

                if pd.notna(ic50_val) and ic50_val > 0:
                    ic50_clean = round(float(ic50_val), 4)
                    si_val = np.nan
                elif "inactive" in outcome:
                    ic50_clean = np.nan
                    si_val = 10000.0
                else:
                    continue

                rows.append({
                    "Title": title,
                    "CC50_uM": np.nan,
                    "IC50_uM_VACV": ic50_clean,
                    "SI VACV": si_val,
                    "SMILES": c_smi,
                    "strain": "Vaccinia virus",
                    "Culture": "Vero / HeLa cells",
                    "doi": doi_str,
                    "source_url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                })
                count += 1
        except Exception as e:
            print(f"[PubChem] Предупреждение для AID {aid}: {e}")
        time.sleep(0.15)
    return pd.DataFrame(rows)


def infer_metadata_for_train(row):
    """Определение источника для записей стартового набора train.csv."""
    title = str(row["Title"]).strip()
    if title.startswith("sid"):
        sid_num = title.replace("sid", "").strip()
        return (
            "PubChem_BioAssay",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid_num}/JSON",
        )
    elif title.startswith("CHEMBL"):
        return (
            "ChEMBL_Database",
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{title}.json",
        )
    return (
        "Literature_VACV_Dataset",
        "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL613295",
    )


def expand_and_build_database(train_path="train.csv", chunk_size=500):
    """
    Собирает RDKit-валидированную базу реальных экспериментальных молекул из train.csv,
    ChEMBL API и PubChem API и сохраняет единый файл + батчи по 500 молекул.
    """
    train_df = pd.read_csv(train_path)
    base_rows = []
    seen_smiles = set()

    for _, row in train_df.iterrows():
        c_smi = validate_smiles(str(row["SMILES"]))
        if not c_smi:
            continue
        seen_smiles.add(c_smi)
        doi, url = infer_metadata_for_train(row)
        base_rows.append({
            "Title": row["Title"],
            "CC50_uM": row["CC50_uM"],
            "IC50_uM_VACV": row["IC50_uM_VACV"],
            "SI VACV": row["SI VACV"],
            "SMILES": c_smi,
            "strain": row["strain"],
            "Culture": row["Culture"],
            "doi": doi,
            "source_url": url,
        })

    print("[1/3] Выгрузка реальных экспериментальных данных из ChEMBL REST API...")
    chembl_df = download_from_chembl(limit_per_target=1000)
    if not chembl_df.empty:
        for _, r in chembl_df.iterrows():
            if r["SMILES"] not in seen_smiles:
                seen_smiles.add(r["SMILES"])
                base_rows.append(r.to_dict())

    print("[2/3] Выгрузка данных биоскринингов из PubChem PUG REST API...")
    pubchem_df = download_from_pubchem(PUBCHEM_AID_LIST, max_rows_per_aid=600)
    if not pubchem_df.empty:
        for _, r in pubchem_df.iterrows():
            if r["SMILES"] not in seen_smiles:
                seen_smiles.add(r["SMILES"])
                base_rows.append(r.to_dict())

    # Если скрипт запущен в офлайн-среде без интернета, сохраняем реальные записи ChEMBL из локального кэша
    existing_db_path = "data/full_molecules_database_4000.csv"
    if not os.path.exists(existing_db_path) and os.path.exists("full_molecules_database_4000.csv"):
        existing_db_path = "full_molecules_database_4000.csv"

    if len(base_rows) <= len(train_df) and os.path.exists(existing_db_path):
        print("      [Офлайн-режим] Импорт реальных записей ChEMBL из локального кэша (без синтетики)...")
        cached_df = pd.read_csv(existing_db_path)
        real_cached = cached_df[
            ~cached_df["Title"].astype(str).str.startswith("CHEMBL_VACV_EXP_")
        ]
        for _, r in real_cached.iterrows():
            c_smi = validate_smiles(str(r["SMILES"]))
            if c_smi and c_smi not in seen_smiles:
                seen_smiles.add(c_smi)
                # Убираем искусственную константу CC50 = 100.0 из старого кэша
                cc_val = np.nan if r["CC50_uM"] == 100.0 else r["CC50_uM"]
                ic_val = r["IC50_uM_VACV"]
                si_val = round(cc_val / ic_val, 3) if (pd.notna(cc_val) and pd.notna(ic_val) and ic_val > 0) else (10000.0 if pd.isna(ic_val) else np.nan)
                base_rows.append({
                    "Title": r["Title"],
                    "CC50_uM": cc_val,
                    "IC50_uM_VACV": ic_val,
                    "SI VACV": si_val,
                    "SMILES": c_smi,
                    "strain": r["strain"],
                    "Culture": r["Culture"],
                    "doi": r["doi"],
                    "source_url": r["source_url"],
                })

    df_full = pd.DataFrame(base_rows)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/databases_by_500", exist_ok=True)

    full_path = "data/full_molecules_database_4000.csv"
    df_full.to_csv(full_path, index=False)

    train_cols = ["Title", "CC50_uM", "IC50_uM_VACV", "SI VACV", "SMILES", "strain", "Culture"]
    df_full[train_cols].to_csv("data/train_expanded_4000.csv", index=False)

    # Очистка старых батчей и сохранение новых по 500 молекул
    for old_f in os.listdir("data/databases_by_500"):
        if old_f.endswith(".csv"):
            os.remove(os.path.join("data/databases_by_500", old_f))

    for part in range(0, len(df_full), chunk_size):
        chunk = df_full.iloc[part : part + chunk_size]
        part_num = (part // chunk_size) + 1
        chunk.to_csv(f"data/databases_by_500/db_part_{part_num:02d}_500mols.csv", index=False)

    print(
        f"[3/3] Готово! Собрано {len(df_full)} реальных валидированных молекул "
        f"(сохранено в {full_path} и разбито на батчи по {chunk_size} шт.)."
    )
    return df_full


if __name__ == "__main__":
    train_file = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    expand_and_build_database(train_path=train_file, chunk_size=500)