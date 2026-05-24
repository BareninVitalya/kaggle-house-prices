# Kaggle House Prices

## Обзор

Проект решает задачу регрессии Kaggle House Prices: Advanced Regression Techniques — предсказание цены продажи дома (`SalePrice`) по табличным признакам недвижимости из Ames, Iowa. Кодовая база организована как Python-пакет `src/` с разделением ответственности между модулями, а эксперименты и демонстрационные сценарии вынесены в ноутбуки и пайплайны обучения.

**Метрика:** root mean squared error / root mean squared log error в рамках кросс-валидации `KFold` (`N_SPLITS=5`, `SEED=42`).

***

## Структура проекта

```text
kaggle-house-prices/
├── src/
│   ├── __init__.py
│   ├── config.py             # Пути, константы, признаки, гиперпараметры моделей
│   ├── data.py               # Загрузка и сохранение данных
│   ├── features.py           # Feature engineering и sklearn-трансформер
│   ├── modeling.py           # Фабрика пайплайнов и моделей
│   ├── evaluate.py           # CV-оценка и вспомогательные функции валидации
│   ├── ensemble.py           # Ансамблирование (Soft, Voting, Stacking)
│   ├── feature_search.py     # OpenFE + отчёты + ablation-анализ
│   ├── logging_utils.py      # Логирование экспериментов
│   ├── nn_model.py           # PyTorch DNN-регрессор
│   ├── openfe_stage.py       # Этап автоматического отбора OpenFE-признаков
│   ├── profiling.py          # Профилирование датасета и важности признаков
│   ├── train_pipeline.py     # Высокоуровневые сценарии запуска экспериментов
│   ├── tuning.py             # Random Search / Optuna / Grid Search
│   └── tuning_objectives.py  # objective-функции для Optuna
│
├── data/
│   ├── raw/                  # train.csv, test.csv (исходные данные Kaggle)
│   └── processed/            # сохранённые обработанные таблицы и submission-файлы
│
├── models/                   # сохранённые модели и артефакты отбора признаков
├── logs/                     # логи экспериментов и tuning-запусков
├── notebooks/                # ноутбуки с EDA и экспериментами
├── output/                   # итоговые submission.csv
│
├── openfe_tmp_data.feather   # временный служебный файл OpenFE
├── requirements.txt          # зависимости проекта
└── README.md
```

***

## Описание модулей

### `config.py` — центральный конфиг проекта

Единственное место, где задаются пути, ключевые константы, списки признаков и дефолтные гиперпараметры моделей. В конфиге зафиксированы `TARGET_COL = "SalePrice"`, `SEED = 42`, `N_SPLITS = 5`, а также директории `data/raw`, `data/processed`, `models`, `logs` и `output`.

**Ключевые переменные:**

| Переменная | Описание |
|---|---|
| `SEED = 42` | Глобальный random seed для воспроизводимости |
| `N_SPLITS = 5` | Количество фолдов в кросс-валидации |
| `TARGET_COL = "SalePrice"` | Целевая переменная |
| `QUALITY_COLS` | Качественные ordinal-признаки типа `ExterQual`, `KitchenQual`, `GarageCond` |
| `LOG_COLS` | Числовые признаки, для которых используется лог-преобразование |
| `NOISE_FEATURES` | Признаки, исключённые как шумные по результатам экспериментов |
| `IDX_TO_DROP` | Индексы наблюдений, удаляемых как проблемные выбросы |
| `NUM_FEATURES` / `CAT_FEATURES` | Финальный набор числовых и категориальных признаков |
| `DEFAULT_*_PARAMS` | Дефолтные гиперпараметры моделей |
| `NUMERIC_AS_CATEGORICAL_MAX_UNIQUE` | Порог, ниже которого числовой признак можно трактовать как категориальный |

***

### `data.py` — загрузка и сохранение данных

Минимальный модуль для чтения исходных CSV и сохранения обработанных таблиц.

| Функция | Что делает |
|---|---|
| `load_train()` | Читает `data/raw/train.csv` → `pd.DataFrame` |
| `load_test()` | Читает `data/raw/test.csv` → `pd.DataFrame` |
| `save_processed(df, name)` | Сохраняет DataFrame в `data/processed/` |

***

### `features.py` — feature engineering

Модуль содержит базовый класс признаков и sklearn-совместимый трансформер для задачи регрессии цены дома. Здесь сосредоточена логика заполнения пропусков, создания бинарных индикаторов наличия объектов и агрегации площадей.

**Иерархия классов:**

```text
HousePricesFeaturesBase     # Базовый класс: общие методы очистки и генерации признаков
└── HousePricesTransformer  # sklearn-совместимый трансформер (Pipeline-ready)
```

**Что делает базовый класс (`HousePricesFeaturesBase`):**

| Признак / операция | Метод | Описание |
|---|---|---|
| Basement imputation | `_impute_basement` | Категории подвала → `"No"`, числовые площади/счётчики → `0` |
| Garage imputation | `_impute_garage` | Аналогичная логика для гаража |
| Masonry veneer imputation | `_impute_masvnr` | Заполнение `MasVnrType`, `MasVnrArea` |
| `HasGarage` | `_add_has_garage_feature` | Индикатор наличия гаража |
| `HasBsmt` | `_add_has_basement_feature` | Индикатор наличия подвала |
| `HasMasVnr` | `_add_has_masvnr_feature` | Индикатор наличия облицовки |
| Other categorical NA | `_impute_other_cat_cols` | Заполнение `Alley`, `FireplaceQu`, `PoolQC`, `Fence`, `MiscFeature` |
| `LotFrontage` imputation | `_impute_lot_frontage` | Медиана по `Neighborhood` |
| `LotShape` grouping | `_lot_shape_transform` | Объединение редких форм участка |
| `TotalSF` | `_add_total_sf` | Суммарная площадь на основе этажей и подвала |

Дополнительно в трансформере используются:
- удаление шумных признаков из `NOISE_FEATURES`;
- работа с ordinal-качеством через `QUALITY_COLS`;
- логарифмирование скошенных признаков из `LOG_COLS`;
- очистка проблемных наблюдений по `IDX_TO_DROP`.

***

### `modeling.py` — фабрика моделей и пайплайнов

Центральный модуль сборки. Все модели строятся как sklearn `Pipeline` из шагов `feat → prep → model`, где `feat` отвечает за ручной feature engineering, `prep` — за табличный препроцессинг, а `model` — за сам регрессор.

**Архитектура пайплайна:**

```text
Pipeline:
  feat  → HousePricesTransformer / FunctionTransformer(identity)
  prep  → ColumnTransformer
  model → регрессор
```

**Ключевые функции:**

| Функция | Описание |
|---|---|
| `get_transformer(...)` | Выбирает трансформер признаков или identity-режим |
| `build_matual_info_preprocessor(X)` | Препроцессор для расчёта mutual information |
| `build_preprocessor(X)` | Общий `ColumnTransformer` для числовых и категориальных признаков |
| `build_catboost_preprocessor(X)` | Препроцессор под CatBoost |
| `build_<name>_model(X, params)` | Фабричные функции для конкретных моделей |
| `build_model(name, X, params)` | Единая фабрика по строковому имени модели |
| `train_model(...)` | Обучение модели на train-датасете |
| `predict_and_save_*` | Генерация submission для Kaggle |

**Поддерживаемые модели:**

| Ключи | Модель |
|---|---|
| `"linear"`, `"linreg"` | LinearRegression |
| `"ridge"` | Ridge |
| `"lasso"` | Lasso |
| `"elastic"` / `"elasticnet"` | ElasticNet |
| `"knn"` | KNeighborsRegressor |
| `"tree"` / `"dt"` | DecisionTreeRegressor |
| `"rf"` | RandomForestRegressor |
| `"catboost"` / `"cat"` | CatBoostRegressor |
| `"lgbm"` / `"lightgbm"` | LGBMRegressor |
| `"xgb"` / `"xgboost"` | XGBRegressor |
| `"dnn"` / `"nn"` / `"mlp"` | `HouseDNNRegressor` |

> Параметр `transform_off=True` позволяет передавать уже подготовленные признаки напрямую в препроцессор, минуя `HousePricesTransformer`. Это удобно для ансамблей, OpenFE и кастомных экспериментов.

***

### `evaluate.py` — кросс-валидация и оценка качества

Модуль с воспроизводимой оценкой моделей. Для регрессионных метрик здесь используется `KFold`, а не `StratifiedKFold`, что соответствует задаче предсказания `SalePrice`.

| Функция | Описание |
|---|---|
| `_get_cv(...)` | Возвращает `KFold` или `StratifiedKFold` в зависимости от метрики |
| `cv_scores(model, X, y, ...)` | Возвращает `mean`, `std`, массив score'ов и при необходимости train-score |
| `cv_scores_with_fit_params(...)` | CV с передачей параметров в `fit` |
| `compare_models(...)` | Сравнение нескольких моделей в едином DataFrame |

***

### `ensemble.py` — ансамблирование моделей

Модуль для объединения нескольких регрессоров.

#### `SoftEnsemble`
Кастомное усреднение предсказаний с поддержкой весов.
- `.build(X)` — собирает базовые модели
- `.fit(X, y)` — обучает все модели
- `.predict(X)` — возвращает усреднённый прогноз
- `.predict_test(...)` — формирует submission

#### `VotingEnsemble`
Обёртка над sklearn `VotingRegressor`, если нужен стандартный sklearn-интерфейс.

#### `StackingEnsembleRegressor`
Внутри модуль использует регрессионный stacking (`StackingRegressor`) и инфраструктуру для meta-model подхода.

***

### `feature_search.py` — автоматический feature engineering (OpenFE)

Модуль для генерации и анализа новых признаков через библиотеку OpenFE.

| Функция | Описание |
|---|---|
| `run_openfe(X_train, y_train, n_features)` | Запускает OpenFE и возвращает топ-N признаков |
| `apply_openfe(X_train, X_test, features)` | Применяет автоматически найденные признаки |
| `save_features(features, path)` | Сохраняет признаки через pickle |
| `load_features(path)` | Загружает сохранённые признаки |
| `feature_importance_report(features, top_n)` | Формирует таблицу важности |
| `ablation_openfe(X_base, X_with_openfe, y, features, step)` | Поэтапная проверка вклада новых признаков |

***

### `logging_utils.py` — логирование экспериментов

Простая система записи результатов экспериментов в JSON-лог без тяжёлых внешних платформ.

| Функция | Описание |
|---|---|
| `log_experiment(name, score, std, params, col_names)` | Записывает результат в лог-файл |
| `load_experiments(logfile)` | Загружает лог как `pd.DataFrame` |

***

### `nn_model.py` — PyTorch DNN-регрессор

Реализация `HouseDNNRegressor` — sklearn-совместимого регрессора на основе полносвязной нейросети. Внутри модуля есть:
- `HouseMLP` — архитектура сети;
- выбор оптимизатора, scheduler и loss function;
- методы `fit`, `_train_epoch`, `_evaluate` для обучения и валидации.

Такой формат позволяет использовать DNN в том же интерфейсе, что и классические sklearn-модели.

***

### `openfe_stage.py` — отбор OpenFE-признаков как отдельный этап пайплайна

Модуль оформляет OpenFE не просто как набор функций, а как полноценный этап отбора:
- преобразование score в единый формат;
- удаление сильно коррелированных признаков;
- жадный отбор лучших кандидатов;
- ablation-анализ прироста качества;
- сохранение результата в виде артефакта.

Главный класс: `OpenFEStage`.

***

### `profiling.py` — профилирование данных

Утилитарный модуль для EDA и диагностики признаков. `DataProfiler` помогает:
- определять типы колонок;
- получать сводку по датасету;
- анализировать числовые, категориальные, булевы и datetime-признаки;
- оценивать взаимосвязи признаков с таргетом;
- использовать permutation importance и другие быстрые диагностические процедуры.

Это особенно полезно в House Prices, где много смешанных по типу признаков и пропусков.

***

### `train_pipeline.py` — сценарии запуска экспериментов

Высокоуровневый модуль для воспроизводимых прогонов.

| Функция | Описание |
|---|---|
| `run_experiment(...)` | Полный запуск обучения и логирования эксперимента |
| `quick_experiment(...)` | Быстрый запуск модели с базовыми настройками |
| `dnn_cv_with_history(...)` | CV-запуск DNN с историей обучения |

Модуль связывает вместе загрузку данных, сборку модели, CV-оценку, логирование и при необходимости OpenFE-этап.

***

### `tuning.py` — подбор гиперпараметров

Модуль для систематического тюнинга моделей.

| Функция | Описание |
|---|---|
| `tune_with_random_search(...)` | Подбор через `RandomizedSearchCV` |
| `tune_with_optuna(...)` | Байесовский поиск через Optuna |
| `tune_dnn_with_grid_search(...)` | Grid search для DNN-сетапов |

В `tune_with_optuna(...)` автоматически подбирается нужный objective-класс под конкретную модель, создаётся `study`, а лучший результат логируется в файл.

***

### `tuning_objectives.py` — objective-функции для Optuna

Содержит отдельные objective-классы под разные модели:
- `LogregObjective`
- `RidgeObjective`
- `LassoObjective`
- `ElasticNetObjective`
- `KNNObjective`
- `TreeObjective`
- `RFObjective`
- `CatBoostObjective`
- `XGBObjective`
- `LGBMObjective`

Такое разбиение удобно тем, что каждая модель получает свой диапазон гиперпараметров и свою логику оптимизации.

***

## Как запустить проект

```bash
# 1. Клонировать репозиторий
git clone https://github.com/BareninVitalya/kaggle-house-prices.git
cd kaggle-house-prices

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Положить данные Kaggle в data/raw/
#    train.csv и test.csv скачиваются с:
#    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# 4. Запустить ноутбук или пайплайн
jupyter notebook notebooks
```

> Для воспроизводимости результатов основные параметры уже зафиксированы в `config.py`: `SEED=42`, `N_SPLITS=5`, `TARGET_COL="SalePrice"`.

****

> **Примечание по Optuna и RMSE**
>
> В новых версиях `scikit-learn` параметр `squared` в `mean_squared_error` помечен как устаревший и вызывает предупреждения/ошибки при использовании Optuna с метрикой RMSE.
>
> Перед запуском Optuna стоит проверить код (включая код самой библиотеки/интеграций) и заменить все вызовы:
>
> ```python
> mean_squared_error(y_true, y_pred, squared=False)
> ```
>
> на:
>
> ```python
> rmse = np.sqrt(mean_squared_error(y_true, y_pred))
> ```
>
> или:
>
> ```python
> rmse = root_mean_squared_error(y_true, y_pred)
> ```
>
> Это устраняет проблему с устаревшим параметром и делает расчёт RMSE совместимым с актуальным `scikit-learn`.