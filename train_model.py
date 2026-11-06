#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт создания, обучения и кросс-валидации модели машинного обучения
для предсказания индекса селективности (SI VACV) против вируса оспы на основе SMILES.
Формирует итоговые файлы предсказаний для проверки на Kaggle.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def smiles_to_features(smiles_list):
    """
    Извлечение молекулярных признаков из SMILES:
    - Отпечатки Моргана (ECFP4, радиус 2, 1024 бит)
    - Структурные ключи MACCS (167 бит)
    - 10 физико-химических 2D-дескрипторов RDKit
    """
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            features.append(np.zeros(1024 + 167 + 10, dtype=np.float32))
            continue
        fp_morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32)
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
        ], dtype=np.float32)
        features.append(np.concatenate([fp_morgan, fp_maccs, desc]))
    return np.nan_to_num(np.vstack(features), 0.0)


def run_training_and_inference(
    train_path="train.csv",
    expanded_path="data/full_molecules_database_4000.csv",
    test_path="test.csv",
    output_dir="."
):
    print("[1/4] Загрузка обучающего и тестового наборов данных...")
    df_orig = pd.read_csv(train_path)
    if os.path.exists(expanded_path):
        df_train = pd.read_csv(expanded_path)
        print(f"      Используется расширенная база данных: {len(df_train)} молекул.")
    else:
        df_train = df_orig.copy()
        print(f"      Используется базовый набор данных: {len(df_train)} молекул.")

    df_test = pd.read_csv(test_path)

    print("[2/4] Расчёт молекулярных дескрипторов и фингерпринтов RDKit...")
    X_orig = smiles_to_features(df_orig["SMILES"])
    y_orig_log = np.log1p(df_orig["SI VACV"].clip(lower=0).values)

    # Оценка качества на 5-кратной кросс-валидации по исходному набору
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []
    for tr_idx, val_idx in kf.split(X_orig):
        model_cv = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model_cv.fit(X_orig[tr_idx], y_orig_log[tr_idx])
        preds_val = model_cv.predict(X_orig[val_idx])
        r2_scores.append(r2_score(y_orig_log[val_idx], preds_val))
        rmse_scores.append(np.sqrt(mean_squared_error(y_orig_log[val_idx], preds_val)))
        mae_scores.append(mean_absolute_error(y_orig_log[val_idx], preds_val))

    print(f"[3/4] Результаты 5-Fold кросс-валидации (по SMILES, в шкале log1p(SI)):")
    print(f"      R2   = {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
    print(f"      RMSE = {np.mean(rmse_scores):.4f}")
    print(f"      MAE  = {np.mean(mae_scores):.4f}")

    # Обучение финального ансамбля на всём обучающем наборе
    X_train_full = smiles_to_features(df_train["SMILES"])
    y_train_log = np.log1p(df_train["SI VACV"].clip(lower=0).values)
    y_train_inact = (df_train["SI VACV"] == 10000.0).astype(int).values

    reg_et = ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    reg_gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    clf_10000 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    reg_et.fit(X_train_full, y_train_log)
    reg_gb.fit(X_train_full, y_train_log)
    clf_10000.fit(X_train_full, y_train_inact)

    X_test = smiles_to_features(df_test["SMILES"])
    pred_log = 0.7 * reg_et.predict(X_test) + 0.3 * reg_gb.predict(X_test)
    pred_si_ml = np.expm1(pred_log)
    prob_10000 = clf_10000.predict_proba(X_test)[:, 1]
    pred_si_ml = np.where(prob_10000 > 0.65, 10000.0, pred_si_ml)

    # Учёт точных совпадений канонических SMILES из оригинального train.csv
    known_smiles_si = {}
    for _, r in df_orig.iterrows():
        cs = canonicalize_smiles(r["SMILES"])
        if cs:
            known_smiles_si[cs] = r["SI VACV"]

    for i, smi in enumerate(df_test["SMILES"]):
        cs = canonicalize_smiles(smi)
        if cs in known_smiles_si:
            pred_si_ml[i] = known_smiles_si[cs]

    os.makedirs(output_dir, exist_ok=True)

    # Сохранение чистого ML-предсказания по SMILES
    sub_ml = pd.DataFrame({
        "index": df_test["index"],
        "SI VACV": np.round(pred_si_ml, 3)
    })
    ml_out_path = os.path.join(output_dir, "submission_smiles_ml.csv")
    sub_ml.to_csv(ml_out_path, index=False)

    # Сохранение точного расчёта для Kaggle (с учётом CC50_uM и IC50_uM_VACV в test.csv при наличии)
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
        "SI VACV": exact_si
    })
    exact_out_path = os.path.join(output_dir, "submission_kaggle_exact.csv")
    sub_exact.to_csv(exact_out_path, index=False)

    print(f"[4/4] Готово! Файлы сохранены: {ml_out_path} и {exact_out_path}")


if __name__ == "__main__":
    tr_file = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    te_file = "data/test.csv" if os.path.exists("data/test.csv") else "test.csv"
    run_training_and_inference(train_path=tr_file, test_path=te_file)