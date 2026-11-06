# Прогнозирование индекса селективности (SI) малых молекул против вируса оспы (VACV)

Репозиторий содержит решение задачи хакатона по разработке модели машинного обучения и автоматизированному сбору базы данных активности малых молекул против вируса осповакцины (Vaccinia virus, штамм VACV).

## Архитектура проекта

```text
├── data/
│   ├── train.csv                         # Исходный обучающий датасет (320 молекул)
│   ├── test.csv                          # Тестовый датасет для проверки (597 молекул)
│   ├── sample_submit.csv                 # Шаблон сабмита Kaggle
│   ├── full_molecules_database_4000.csv  # Собранная единая база (4000 молекул с DOI и ссылками)
│   ├── train_expanded_4000.csv           # Расширенный датасет в формате train.csv
│   └── databases_by_500/                 # Разбивка собранной базы на 8 файлов по 500 молекул
│       ├── db_part_01_500mols.csv
│       └── ...
├── fetch_data.py                         # Скрипт автоматической выгрузки и RDKit-валидации данных из ChEMBL / PubChem
├── train_model.py                        # Скрипт генерации дескрипторов, обучения ML-модели и кросс-валидации
├── run_pipeline.py                       # Единый воспроизводимый пайплайн (запуск от выгрузки до сабмита Kaggle)
├── submission_kaggle_exact.csv           # Итоговый файл предсказаний для Kaggle
├── submission_smiles_ml.csv              # Предсказания ансамбля моделей только по SMILES
├── requirements.txt                      # Зависимости проекта
└── README.md                             # Документация проекта
```

## Источники данных
Выгрузка данных и метаданных (включая `DOI` публикаций) осуществляется из открытых API:
- **ChEMBL REST API**: `https://www.ebi.ac.uk/chembl/api/data/activity.json` (мишени `CHEMBL613295` — Vaccinia virus, `CHEMBL613120` — Monkeypox virus, `CHEMBL613728` — Cowpox virus, `CHEMBL612745` — Variola virus, `CHEMBL613729` — Ectromelia virus).
- **PubChem PUG REST API**: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/` (биоскрининги антивирусной активности).

Все структуры проходят проверку валентности, санитизацию и приведение к каноническому SMILES через `RDKit`.

## Быстрый запуск (воспроизводимость пайплайна)

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите полный пайплайн одной командой:
```bash
python run_pipeline.py
```

Либо по шагам:
- Только сбор и валидация базы данных: `python fetch_data.py`
- Только обучение модели и генерация `.csv` для Kaggle: `python train_model.py`