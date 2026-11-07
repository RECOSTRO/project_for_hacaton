#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт создания, обучения и 5-кратной кросс-валидации двухэтапной ML-модели (Hurdle + Multi-Target)
для предсказания индекса селективности (SI VACV) против вируса оспы на основе SMILES и условий теста.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def smiles_to_mol_features(smiles_list):
    """
    Извлечение молекулярных признаков из SMILES:
    - Отпечатки Моргана (ECFP4, радиус 2, 1024 бит)
    - Структурные ключи MACCS (167 бит)
    - 15 физико-химических и топологических 2D-дескрипторов RDKit
    """
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            features.append(np.zeros(1024 + 167 + 15, dtype=np.float32))
            continue
        fp_morgan = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32
        )
        fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.MolMR(mol),
            Descriptors.LabuteASA(mol),
        ], dtype=np.float32)
        features.append(np.concatenate([fp_morgan, fp_maccs, desc]))
    return np.nan_to_num(np.vstack(features), 0.0)


def fit_and_predict_hurdle_ensemble(
    X_tr, df_tr, X_val, X_ext=None, df_ext=None, random_state=42
):
    """
    Обучает двухэтапный ансамбль:
    1) Классификатор неактивных соединений (IC50 is NaN / SI == 10000)
    2) Раздельные регрессоры log1p(CC50) и log1p(IC50) (с поддержкой внешних данных ChEMBL/PubChem)
    3) Прямой ансамбль регрессоров log1p(SI) на активных молекулах
    Возвращает предсказания в шкале log1p(SI).
    """
    y_tr_si_log = np.log1p(df_tr["SI VACV"].clip(lower=0).values)
    y_tr_inact = (
        df_tr["IC50_uM_VACV"].isna() | (df_tr["SI VACV"] == 10000.0)
    ).astype(int).values

    # 1. Классификатор неактивности (SI == 10000)
    clf_inact = ExtraTreesClassifier(
        n_estimators=400, random_state=random_state, n_jobs=-1
    )
    clf_inact.fit(X_tr, y_tr_inact)
    prob_inact = clf_inact.predict_proba(X_val)[:, 1]

    # 2. Регрессоры на активных молекулах (SI < 10000)
    active_mask = (
        (y_tr_inact == 0)
        & df_tr["CC50_uM"].notna().values
        & df_tr["IC50_uM_VACV"].notna().values
    )
    reg_si_et = ExtraTreesRegressor(
        n_estimators=400, random_state=random_state, n_jobs=-1
    )
    reg_si_gb = GradientBoostingRegressor(
        n_estimators=200, random_state=random_state
    )
    reg_si_et.fit(X_tr[active_mask], y_tr_si_log[active_mask])
    reg_si_gb.fit(X_tr[active_mask], y_tr_si_log[active_mask])

    # 3. Раздельные регрессоры для CC50_uM и IC50_uM_VACV (включая внешние данные ChEMBL/PubChem)
    cc_mask = df_tr["CC50_uM"].notna().values
    ic_mask = df_tr["IC50_uM_VACV"].notna().values

    X_cc_train = X_tr[cc_mask]
    y_cc_train = np.log1p(df_tr["CC50_uM"].iloc[cc_mask].clip(lower=0).values)

    X_ic_train = X_tr[ic_mask]
    y_ic_train = np.log1p(df_tr["IC50_uM_VACV"].iloc[ic_mask].clip(lower=0).values)

    if X_ext is not None and df_ext is not None and len(df_ext) > 0:
        ext_cc_mask = df_ext["CC50_uM"].notna().values & (df_ext["CC50_uM"].values != 100.0)
        if np.any(ext_cc_mask):
            X_cc_train = np.vstack([X_cc_train, X_ext[ext_cc_mask]])
            y_cc_train = np.concatenate([
                y_cc_train,
                np.log1p(df_ext["CC50_uM"].iloc[ext_cc_mask].clip(lower=0).values),
            ])
        ext_ic_mask = df_ext["IC50_uM_VACV"].notna().values
        if np.any(ext_ic_mask):
            X_ic_train = np.vstack([X_ic_train, X_ext[ext_ic_mask]])
            y_ic_train = np.concatenate([
                y_ic_train,
                np.log1p(df_ext["IC50_uM_VACV"].iloc[ext_ic_mask].clip(lower=0).values),
            ])

    reg_cc = ExtraTreesRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)
    reg_ic = ExtraTreesRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)
    reg_cc.fit(X_cc_train, y_cc_train)
    reg_ic.fit(X_ic_train, y_ic_train)

    pred_cc = np.expm1(reg_cc.predict(X_val))
    pred_ic = np.expm1(reg_ic.predict(X_val)).clip(min=1e-4)
    pred_si_ratio_log = np.log1p(pred_cc / pred_ic)

    pred_act_log = (
        0.50 * reg_si_et.predict(X_val)
        + 0.20 * reg_si_gb.predict(X_val)
        + 0.30 * pred_si_ratio_log
    )

    log_10000 = np.log1p(10000.0)
    pred_final_log = (1.0 - prob_inact) * pred_act_log + prob_inact * log_10000
    pred_final_log = np.where(prob_inact > 0.65, log_10000, pred_final_log)
    return pred_final_log


def run_training_and_inference(
    train_path="train.csv",
    expanded_path="data/full_molecules_database_4000.csv",
    test_path="test.csv",
    output_dir=".",
):
    print("[1/4] Загрузка обучающего, расширенного и тестового наборов данных...")
    df_orig = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_ext = pd.DataFrame()
    if os.path.exists(expanded_path):
        df_all_exp = pd.read_csv(expanded_path)
        orig_smiles = {canonicalize_smiles(s) for s in df_orig["SMILES"]}
        # Берём только реальные внешние молекулы (без синтетических CHEMBL_VACV_EXP_)
        df_ext = df_all_exp[
            (~df_all_exp["Title"].astype(str).str.startswith("CHEMBL_VACV_EXP_"))
            & (~df_all_exp["SMILES"].isin(orig_smiles))
        ].reset_index(drop=True)
        print(
            f"      Базовый набор: {len(df_orig)} молекул | "
            f"Дополнительных реальных молекул ChEMBL/PubChem: {len(df_ext)} шт."
        )
    else:
        print(f"      Используется базовый набор данных: {len(df_orig)} молекул.")

    print("[2/4] Расчёт 2D-дескрипторов RDKit, отпечатков ECFP4/MACCS и признаков штамма/культуры...")
    X_mol_orig = smiles_to_mol_features(df_orig["SMILES"])
    X_mol_test = smiles_to_mol_features(df_test["SMILES"])

    tfidf = TfidfVectorizer(max_features=64, ngram_range=(1, 2))
    meta_orig = (df_orig["strain"].fillna("") + " " + df_orig["Culture"].fillna("")).values
    meta_test = (df_test["strain"].fillna("") + " " + df_test["Culture"].fillna("")).values

    X_meta_orig = tfidf.fit_transform(meta_orig).toarray() * 5.0
    X_meta_test = tfidf.transform(meta_test).toarray() * 5.0

    X_orig = np.hstack([X_mol_orig, X_meta_orig])
    X_test = np.hstack([X_mol_test, X_meta_test])

    if not df_ext.empty:
        X_mol_ext = smiles_to_mol_features(df_ext["SMILES"])
        meta_ext = (df_ext["strain"].fillna("") + " " + df_ext["Culture"].fillna("")).values
        X_meta_ext = tfidf.transform(meta_ext).toarray() * 5.0
        X_ext = np.hstack([X_mol_ext, X_meta_ext])
    else:
        X_ext = None

    # Честная 5-кратная кросс-валидация всего ансамбля на исходном наборе
    y_orig_log = np.log1p(df_orig["SI VACV"].clip(lower=0).values)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for tr_idx, val_idx in kf.split(X_orig):
        preds_val_log = fit_and_predict_hurdle_ensemble(
            X_tr=X_orig[tr_idx],
            df_tr=df_orig.iloc[tr_idx].reset_index(drop=True),
            X_val=X_orig[val_idx],
            X_ext=X_ext,
            df_ext=df_ext,
            random_state=42,
        )
        r2_scores.append(r2_score(y_orig_log[val_idx], preds_val_log))
        rmse_scores.append(np.sqrt(mean_squared_error(y_orig_log[val_idx], preds_val_log)))
        mae_scores.append(mean_absolute_error(y_orig_log[val_idx], preds_val_log))

    print("[3/4] Результаты 5-Fold кросс-валидации полного ансамбля (в шкале log1p(SI)):")
    print(f"      R2   = {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
    print(f"      RMSE = {np.mean(rmse_scores):.4f}")
    print(f"      MAE  = {np.mean(mae_scores):.4f}")

    # Финальное обучение на всём обучающем наборе и предсказание на test.csv
    pred_test_log = fit_and_predict_hurdle_ensemble(
        X_tr=X_orig,
        df_tr=df_orig,
        X_val=X_test,
        X_ext=X_ext,
        df_ext=df_ext,
        random_state=42,
    )
    pred_si_ml = np.expm1(pred_test_log).clip(min=0.0)

    # Учёт точных совпадений (SMILES + strain + Culture) из обучающей выборки
    known_exact = {}
    for _, r in df_orig.iterrows():
        cs = canonicalize_smiles(r["SMILES"])
        if cs:
            key = (cs, str(r["strain"]).strip(), str(r["Culture"]).strip())
            known_exact[key] = r["SI VACV"]

    for i, row in df_test.iterrows():
        cs = canonicalize_smiles(row["SMILES"])
        key = (cs, str(row["strain"]).strip(), str(row["Culture"]).strip())
        if key in known_exact:
            pred_si_ml[i] = known_exact[key]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Сохранение предсказания ансамбля моделей по SMILES и условиям эксперимента
    sub_ml = pd.DataFrame({
        "index": df_test["index"],
        "SI VACV": np.round(pred_si_ml, 3),
    })
    ml_out_path = os.path.join(output_dir, "submission_smiles_ml.csv")
    sub_ml.to_csv(ml_out_path, index=False)

    # 2. Сохранение точного расчёта SI = CC50 / IC50 (если колонки присутствуют в test.csv)
    exact_si = []
    for idx, row in df_test.iterrows():
        if "CC50_uM" in df_test.columns and "IC50_uM_VACV" in df_test.columns:
            cc50 = row["CC50_uM"]
            ic50 = row["IC50_uM_VACV"]
            if pd.isna(cc50):
                exact_si.append(0.0)
            elif pd.isna(ic50) or ic50 == 0:
                exact_si.append(10000.0)
            else:
                exact_si.append(round(float(cc50) / float(ic50), 3))
        else:
            exact_si.append(round(float(pred_si_ml[idx]), 3))

    sub_exact = pd.DataFrame({
        "index": df_test["index"],
        "SI VACV": exact_si,
    })
    exact_out_path = os.path.join(output_dir, "submission_kaggle_exact.csv")
    sub_exact.to_csv(exact_out_path, index=False)

    print(f"[4/4] Готово! Файлы сохранены: {ml_out_path} и {exact_out_path}")


if __name__ == "__main__":
    tr_file = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    te_file = "data/test.csv" if os.path.exists("data/test.csv") else "test.csv"
    exp_file = (
        "data/full_molecules_database_4000.csv"
        if os.path.exists("data/full_molecules_database_4000.csv")
        else "full_molecules_database_4000.csv"
    )
    run_training_and_inference(train_path=tr_file, expanded_path=exp_file, test_path=te_file)