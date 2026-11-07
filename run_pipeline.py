#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from fetch_data import expand_and_build_database
from train_model import run_training_and_inference

if __name__ == "__main__":
    tr_path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    te_path = "data/test.csv" if os.path.exists("data/test.csv") else "test.csv"
    db_path = "data/full_molecules_database_4000.csv"

    print("=== ШАГ 1: Проверка и сбор базы данных молекул (ChEMBL + PubChem) ===")
    if os.path.exists(db_path):
        print(f"      База {db_path} уже существует, пропускаем повторную выгрузку из API.")
    else:
        expand_and_build_database(train_path=tr_path, chunk_size=500)

    print("\n=== ШАГ 2: Обучение ансамбля моделей и генерация сабмитов для Kaggle ===")
    run_training_and_inference(
        train_path=tr_path,
        expanded_path=db_path,
        test_path=te_path,
        output_dir=".",
    )
    print("\n=== Пайплайн успешно выполнен! ===")