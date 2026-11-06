#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый воспроизводимый пайплайн для хакатона (согласно требованиям жюри):
1. Выгружает и валидирует базу данных молекул (fetch_data.py)
2. Обучает ML-модель и генерирует предсказания для Kaggle (train_model.py)
"""

import os
from fetch_data import expand_and_build_database
from train_model import run_training_and_inference

if __name__ == "__main__":
    print("=== ШАГ 1: Сбор и валидация базы данных молекул ===")
    tr_path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    te_path = "data/test.csv" if os.path.exists("data/test.csv") else "test.csv"

    expand_and_build_database(train_path=tr_path, target_total=4000, chunk_size=500)

    print("\n=== ШАГ 2: Обучение модели и генерация сабмитов для Kaggle ===")
    run_training_and_inference(
        train_path=tr_path,
        expanded_path="data/full_molecules_database_4000.csv",
        test_path=te_path,
        output_dir="."
    )
    print("\n=== Пайплайн успешно выполнен! ===")