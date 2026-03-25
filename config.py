from typing import Dict

LOCAL_ITERATIONS_PER_GENERATION: int = 2 # Количество локальных итераций 
LOCAL_CANDIDATES_PER_ITERATION: int = 3  # Сколько градиентов генерируем за одну итерацию
LOCAL_BATCH_SIZE: int = 20               # Максимальное число failure-примеров в одном батче
LOCAL_MAX_EXAMPLES: int = 15             # Максимальное число примеров, передаваемых в один LLM вызов
LOCAL_PARENTS_PER_ITERATION: int = 3     # Сколько лучших узлов поколения использовать как "родителей" для генерации новых кандидатов
GLOBAL_CANDIDATES: int = 4               # Ширина глобального поиска
GLOBAL_TRIGGER_INTERVAL: int = 1         # Каждые N поколений
GLOBAL_HISTORY_WINDOW: int = 15          # Сколько узлов истории анализировать
MAX_GENERATIONS: int = 2                 # Максимальное число поколений
POPULATION_SIZE: int = 3                 # Сколько промптов держим в активном пуле
PATIENCE: int = 2                        # Поколений без улучшения до остановки
MIN_IMPROVEMENT: float = 0.015           # Минимальное улучшение для продолжения
DIVERSITY_BONUS: float = 0.04            # Бонус за разнообразие
SIMILARITY_THRESHOLD: float = 0.80       # Порог схожести
TEMPERATURE: float = 0.1                 # Температура для LLM
MAX_TOKENS: int = 3000                   # Максимальное число токенов
PROVIDER = "openai"                      # Провайдер LLM
API_KEY = ""                             # Ключ API провайдера
MODEL = "gpt-3.5-turbo"                  # Модель LLM
MIN_PROMPT_LENGTH = 20                   # Минимальная длина текста, считающегося валидным промптом
FALLBACK_ANALYSIS_LENGTH = 500           # Сколько символов взять из ответа, если секция ERROR ANALYSIS отсутствует
DEFAULT_PRIORITY = 0.5                   # Приоритет градиента, если LLM не указал его
MIN_LIST_ITEM_LENGTH = 5                 # Порог минимальной длины элемента списка в ответе LLM
LINEAGE_RECENT_OPS_LIMIT = 3             # Сколько последних операций включать
MAX_SUCCESS_EXAMPLES = 5                 # Верхний предел успешных примеров в анализе
SUCCESS_EXAMPLE_LIMIT = 5                # Ограничение на success-примеры при fallback-генерации
CONTRASTIVE_PRIORITY_BOOST = 0.1         # Бонус к приоритету контрастного градиента
FAILURE_EXAMPLE_LIMIT = 5                # Минимальное число failure-примеров, чтобы запускать кластеризацию
BATCH_SUCCESS_EXAMPLE_LIMIT = 5          # Лимит успешных примеров при пакетной генерации градиентов
CLUSTERING_FAILURE_MULTIPLIER = 2        # Во сколько раз число провалов должно превышать batch size, чтобы включить кластеризацию
MAX_CONTEXT_OPERATIONS = 5               # Сколько успешных операций из истории передавать в контекст LLM
MIN_EXAMPLES_FOR_CONTRASTIVE = 5         # Минимум успехов и провалов для контрастного градиента
DEFAULT_PARETO_METRICS = ("accuracy", "safety", "robustness") # Набор метрик, по которым строится Pareto-front
DEFAULT_STAGNATION_WINDOW = 5            # Размер окна поколений для определения застоя
TOP_BEST_NODES: int = 5                  # Количество лучших узлов
MAX_DISTANCE_PAIRS: int = 10             # Максимальное число пар узлов для вычисления попарных метрик
COMMON_WORDS_TOP_K: int = 20             # Сколько наиболее частых слов учитывать при извлечении «общих фраз» из лучших промптов
COMMON_WORD_MIN_FREQ: int = 3            # Минимальная частота слова, чтобы оно считалось при извлечении «общих фраз»
FAILED_PERCENTILE: int = 20              # Процентиль для определения «плохих» промптов
FAILED_OP_MIN_COUNT: int = 3             # Минимальное количество раз, которое операция должна встретиться среди неудачных узлов, чтобы считаться «неудачным направлением»
MIN_GLOBAL_SOURCE_USAGE: int = 3         # Минимальное количество глобальных изменений, чтобы не считать глобальные стратегии недостаточно применёнными
STAGNATION_SIMILARITY_THRESHOLD: float = 0.7 # Порог сходства между лучшими узлами, выше которого считается, что оптимизация зашла в застой
DIVERSITY_DISTANCE_THRESHOLD: float = 0.3 # Порог среднего расстояния между узлами для оценки, нужна ли диверсификация
LOW_DIVERSITY_THRESHOLD: float = 0.2     # Порог низкого разнообразия, ниже которого запускается глобальный шаг
MAX_DIVERSITY_SAMPLES: int = 5           # Максимальное число узлов, используемых для оценки разнообразия, чтобы не считать все пары
MIN_NODES_FOR_DIVERSITY: int = 3         # Минимальное число узлов поколения, чтобы проверять разнообразие
RECENT_GENERATIONS_FOR_DIVERSITY = 4     # Количество последних поколений, которые используются для оценки разнообразия популяции промптов
MAX_EXAMPLES_PER_NODE: int = 100         # Максимальное число примеров на узел
GLOBAL_OPT_AVG_PATH_LENGTH: int = 5      # Средняя длина пути в паттернах глобальной оптимизации
BATCH_EVAL_SIZE: int = 25                # Размер батча для групповых LLM-запросов при оценке
LLM_CACHE_ENABLED: bool = True           # Включить кэширование ответов LLM в памяти
CACHE_MAX_SIZE: int = 10000              # Максимальное число элементов в LRU-кэше
LLM_PERSISTENT_CACHE: bool = True        # Включить persistent SQLite cache
CACHE_DB_PATH: str = "optimization_results/llm_cache.sqlite"  # Путь к sqlite DB
CACHE_TTL_SECONDS: int = 7 * 24 * 3600   # TTL для записей в кэше (по умолчанию 7 дней)
USE_LLM_CORRECTNESS_CHECK: bool = False  # Использовать LLM для проверки корректности (доп. запросы)
USE_LLM_EDIT_DISTANCE: bool = False      # Использовать LLM для семантической дистанции (доп. запросы)
LOCAL_SIMILARITY_THRESHOLD: float = 0.8  # Порог локальной семантической близости (0..1)
ENABLE_DIAGNOSTIC_LOGS: bool = True      # Включить диагностическое логирование
EXEMPLAR_COUNT: int = 5                  # Число wrong-exemplars в мета-промпте
HISTORY_SCORE_THRESHOLD: float = 0.4     # 0.0 — все инструкции; например 0.4 отрежет нижние 40%
# Стратегия выбора wrong-exemplars для мета-промпта.
# "accumulative_most_frequent" — топ-K по накопленному счётчику провалов по всей истории (default)
# "current_most_frequent"      — топ-K по счётчику провалов среди инструкций текущего мета-промпта
# "random"                     — случайная выборка, seed=current_generation (меняется каждый шаг)
# "constant"                   — фиксированная случайная выборка, seed=0 (всегда одинакова)
EXEMPLAR_SELECTION_STRATEGY: str = "accumulative_most_frequent"
MINI_BATCH_RATIO: float = 0.2            # Доля валидационных примеров для предварительного отбора
PRE_SCREEN_TOP_K: int = 4                # Сколько кандидатов полностью оценивать после предварительного отбора
GRADIENT_MOMENTUM: float = 0.3           # Вес исторического импульса успешных градиентов
CROSSOVER_CANDIDATES: int = 2            # Количество кандидатов-кроссоверов на глобальный шаг
DIVERSITY_WEIGHT: float = 0.08           # Вес бонуса за разнообразие при отборе beam/популяции (для устойчивости к коллапсу beam)
MAX_GRADIENT_PAIRS: int = 8              # Макс. накопленных пар градиент-родитель за одну локальную итерацию
GLOBAL_QUALITY_GATE_TOLERANCE: float = 0.90 # Допуск порога качества для глобального исследования (доля от базовой точности)

# Веса метрик для оценки промптов
METRIC_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.65,
    "safety": 0.02,
    "robustness": 0.05,
    "efficiency": 0.03,
    "f1": 0.25,
}
