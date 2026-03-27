from typing import Dict

LOCAL_ITERATIONS_PER_GENERATION: int = 2 # Количество локальных итераций 
LOCAL_CANDIDATES_PER_ITERATION: int = 2  # Сколько градиентов генерируем за одну итерацию
LOCAL_BATCH_SIZE: int = 20               # Максимальное число failure-примеров в одном батче
LOCAL_MAX_EXAMPLES: int = 50             # Максимальное число примеров, передаваемых в один LLM-вызов
LOCAL_PARENTS_PER_ITERATION: int = 2     # Сколько лучших узлов использовать как «родителей»
MAX_GRADIENT_PAIRS: int = 4              # Макс. пар градиент-родитель за одну локальную итерацию
MINI_BATCH_RATIO: float = 0.5            # Доля валидационных примеров для предварительного отбора
PRE_SCREEN_TOP_K: int = 3                # Сколько кандидатов полностью оценивать после предварительного отбора
SUCCESS_EXAMPLE_LIMIT: int = 5           # Лимит успешных примеров при генерации градиентов
FAILURE_EXAMPLE_LIMIT: int = 5           # Минимальное число failure-примеров для запуска кластеризации

GLOBAL_CANDIDATES: int = 4               # Ширина глобального поиска
GLOBAL_TRIGGER_INTERVAL: int = 2         # Каждые N поколений
GLOBAL_HISTORY_WINDOW: int = 25          # Сколько узлов истории анализировать
GLOBAL_QUALITY_GATE_TOLERANCE: float = 0.85 # Допуск порога качества для глобального исследования
GLOBAL_REFINE_WITH_LOCAL: bool = True    # Проводить ли локальную доработку глобальных кандидатов
GLOBAL_PRESCREEN_GATE: float = 0.7       # Глобальные кандидаты должны набрать >= этой доли от лучшего
GLOBAL_OPTIMIZER_TEMPERATURE: float = 0.7 # Температура LLM для глобального оптимизатора
CROSSOVER_CANDIDATES: int = 2            # Количество кандидатов-кроссоверов на глобальный шаг
EXEMPLAR_COUNT: int = 5                  # Число wrong-exemplars в мета-промпте
FEW_SHOT_COUNT: int = 5                  # Число few-shot примеров в мета-промпте
HISTORY_SCORE_THRESHOLD: float = 0.4     # 0.0 — все инструкции; 0.4 отрежет нижние 40 %
# Стратегия выбора wrong-exemplars для мета-промпта:
# "accumulative_most_frequent" — топ-K по накопленному счётчику провалов (default)
# "current_most_frequent"      — топ-K по счётчику провалов среди инструкций мета-промпта
# "random"                     — случайная выборка, seed = current_generation
# "constant"                   — фиксированная случайная выборка, seed = 0
EXEMPLAR_SELECTION_STRATEGY: str = "accumulative_most_frequent"

GLOBAL_OPT_AVG_PATH_LENGTH: int = 5      # Средняя длина пути в паттернах глобальной оптимизации
RECENT_GENERATIONS_FOR_DIVERSITY: int = 4 # Поколений для оценки разнообразия популяции
DEFAULT_PARETO_METRICS = ("accuracy", "f1", "safety", "robustness", "efficiency") # Метрики Pareto-front
DEFAULT_STAGNATION_WINDOW: int = 5       # Размер окна поколений для определения застоя
TOP_BEST_NODES: int = 5                  # Количество лучших узлов
MAX_DISTANCE_PAIRS: int = 10             # Максимальное число пар для попарных метрик
COMMON_WORDS_TOP_K: int = 20             # Наиболее частых слов при извлечении «общих фраз»
COMMON_WORD_MIN_FREQ: int = 3            # Минимальная частота слова для «общих фраз»
FAILED_PERCENTILE: int = 20              # Процентиль для определения «плохих» промптов
FAILED_OP_MIN_COUNT: int = 3             # Мин. частота операции среди неудачных узлов
MIN_GLOBAL_SOURCE_USAGE: int = 3         # Мин. количество глобальных изменений
STAGNATION_SIMILARITY_THRESHOLD: float = 0.7 # Порог сходства для диагностики застоя
DIVERSITY_DISTANCE_THRESHOLD: float = 0.3 # Порог среднего расстояния для диверсификации
LOW_DIVERSITY_THRESHOLD: float = 0.2     # Ниже — запускается глобальный шаг
MAX_DIVERSITY_SAMPLES: int = 5           # Макс. узлов для оценки разнообразия
MIN_NODES_FOR_DIVERSITY: int = 3         # Мин. узлов поколения для проверки разнообразия

MAX_GENERATIONS: int = 12                # Максимальное число поколений
POPULATION_SIZE: int = 2                 # Сколько промптов держим в активном пуле
PATIENCE: int = 2                        # Поколений без улучшения до остановки
MIN_IMPROVEMENT: float = 0.005           # Минимальное улучшение для продолжения
SIMILARITY_THRESHOLD: float = 0.80       # Порог схожести промптов
DIVERSITY_WEIGHT: float = 0.08           # Бонус за разнообразие при отборе beam/популяции
REFLECTION_ENABLED: bool = True          # Межпоколенческая рефлексия
STAGE3_TOP_K: int = 3                    # Сколько лучших кандидатов получают Stage 3

TEMPERATURE: float = 0.3                 # Температура для LLM
MAX_TOKENS: int = 3000                   # Максимальное число токенов
PROVIDER = "openai"                      # Провайдер LLM
API_KEY = ""                             # Ключ API провайдера
MODEL = "gpt-3.5-turbo"                  # Модель LLM
LLM_CACHE_ENABLED: bool = True           # Кэширование ответов LLM в памяти
CACHE_MAX_SIZE: int = 10000              # Максимальное число элементов в LRU-кэше
LLM_PERSISTENT_CACHE: bool = True        # Persistent SQLite cache
CACHE_DB_PATH: str = "optimization_results/llm_cache.sqlite"
CACHE_TTL_SECONDS: int = 7 * 24 * 3600   # TTL записей кэша (7 дней)

MAX_EXAMPLES_PER_NODE: int = 100         # Максимальное число примеров на узел
BATCH_EVAL_SIZE: int = 50                # Размер батча для групповых LLM-запросов при оценке
JUDGE_BATCH_SIZE: int = 15               # Размер батча для LLM-judge (stage 3)
NORMALIZE_STAGE_WEIGHTS: bool = True     # Нормализовать веса метрик, чтобы composite ∈ [0,1]
USE_LLM_CORRECTNESS_CHECK: bool = False  # LLM для проверки корректности
USE_LLM_EDIT_DISTANCE: bool = False      # LLM для семантической дистанции
CORRECTNESS_TOKEN_F1_THRESHOLD: float = 0.5  # Порог token-F1 для определения правильности
USE_CONTAINMENT_CHECK: bool = True       # Проверять вхождение expected в actual
STABILITY_LAMBDA: float = 0.1            # Штраф за нестабильность: score = mean − λ·std
BOOTSTRAP_RUNS: int = 0                  # Bootstrap-выборки для стабильности (0 = выкл.)
BOOTSTRAP_SAMPLE_RATIO: float = 0.7      # Размер bootstrap-выборки как доля от полной

MIN_PROMPT_LENGTH: int = 20              # Минимальная длина валидного промпта
FALLBACK_ANALYSIS_LENGTH: int = 500      # Символов для fallback, если секция ERROR ANALYSIS пуста
DEFAULT_PRIORITY: float = 0.5            # Приоритет градиента по умолчанию
MIN_LIST_ITEM_LENGTH: int = 5            # Мин. длина элемента списка в ответе LLM
LINEAGE_RECENT_OPS_LIMIT: int = 5        # Сколько последних операций включать в сводку

ENABLE_DIAGNOSTIC_LOGS: bool = True      # Включить диагностическое логирование

from typing import List as _List

METRICS_CONFIG: _List[Dict[str, any]] = [
    {"name": "accuracy",    "weight": 0.30, "stage": 1},  # Accuracy  (exact match + token-F1 containment)
    {"name": "token_f1",    "weight": 0.20, "stage": 1},  # Token F1  (лексическое совпадение)
    {"name": "f1",          "weight": 0.15, "stage": 3},  # F1        (семантическое совпадение)
    {"name": "safety",      "weight": 0.10, "stage": 3},  # Safety    (безопасность ответа)
    {"name": "robustness",  "weight": 0.10, "stage": 3},  # Robustness (устойчивость)
    {"name": "efficiency",  "weight": 0.15, "stage": 3},  # Efficiency (лаконичность)
]

METRIC_WEIGHTS: Dict[str, float] = {m["name"]: m["weight"] for m in METRICS_CONFIG}
