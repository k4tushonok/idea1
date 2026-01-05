from typing import Dict

LOCAL_ITERATIONS_PER_GENERATION: int = 3 # Количество локальных итераций 
LOCAL_CANDIDATES_PER_ITERATION: int = 2  # Сколько градиентов генерируем за одну итерацию
LOCAL_BATCH_SIZE: int = 5                # Максимальное число failure-примеров в одном батче
LOCAL_MAX_EXAMPLES: int = 10             # Максимальное число примеров, передаваемых в один LLM вызов
GLOBAL_CANDIDATES: int = 2               # Ширина глобального поиска
GLOBAL_TRIGGER_INTERVAL: int = 2         # Каждые N поколений
GLOBAL_HISTORY_WINDOW: int = 15          # Сколько узлов истории анализировать
MAX_GENERATIONS: int = 3                 # Максимальное число поколений
POPULATION_SIZE: int = 2                 # Сколько промптов держим в активном пуле
PATIENCE: int = 2                        # Поколений без улучшения до остановки
MIN_IMPROVEMENT: float = 0.005           # Минимальное улучшение для продолжения
DIVERSITY_BONUS: float = 0.03            # Бонус за разнообразие
SIMILARITY_THRESHOLD: float = 0.85       # Порог схожести
TEMPERATURE: float = 0.7                 # Температура для LLM
MAX_TOKENS: int = 2000                   # Максимальное число токенов
PROVIDER = "openai"                      # Провайдер LLM
API_KEY = ""                             # Ключ API провайдера
MODEL = "gpt-4o-mini"                    # Модель LLM
MIN_PROMPT_LENGTH = 20                   # Минимальная длина текста, считающегося валидным промптом
FALLBACK_ANALYSIS_LENGTH = 500           # Сколько символов взять из ответа, если секция ERROR ANALYSIS отсутствует
DEFAULT_PRIORITY = 0.5                   # Приоритет градиента, если LLM не указал его
MIN_LIST_ITEM_LENGTH = 5                 # Порог минимальной длины элемента списка в ответе LLM
LINEAGE_RECENT_OPS_LIMIT = 3             # Сколько последних операций включать
MAX_SUCCESS_EXAMPLES = 5                 # Верхний предел успешных примеров в анализе
SUCCESS_EXAMPLE_LIMIT = 5                # Ограничение на success-примеры при fallback-генерации
CONTRASTIVAE_PRIORITY_BOOST = 0.1        # Бонус к приоритету контрастного градиента
FAILURE_EXAMPLE_LIMIT = 5                # Минимальное число failure-примеров, чтобы запускать кластеризацию
BATCH_SUCCESS_EXAMPLE_LIMIT = 5          # Лимит успешных примеров при пакетной генерации градиентов
MAX_PROMPTS_TO_COMBINE = 3               # Максимальное число промптов для комбинирования в одной операции
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
MIN_OPERATION_USAGE: int = 2             # Минимальное число использования операции в истории, чтобы её не считать «неисследованной»
MIN_GLOBAL_SOURCE_USAGE: int = 3         # Минимальное количество глобальных изменений, чтобы не считать глобальные стратегии недостаточно применёнными
STAGNATION_SIMILARITY_THRESHOLD: float = 0.7 # Порог сходства между лучшими узлами, выше которого считается, что оптимизация зашла в застой
DIVERSITY_DISTANCE_THRESHOLD: float = 0.3 # Порог среднего расстояния между узлами для оценки, нужна ли диверсификация
LOW_DIVERSITY_THRESHOLD: float = 0.2     # Порог низкого разнообразия, ниже которого запускается глобальный шаг
MAX_DIVERSITY_SAMPLES: int = 5           # Максимальное число узлов, используемых для оценки разнообразия, чтобы не считать все пары
MIN_NODES_FOR_DIVERSITY: int = 3         # Минимальное число узлов поколения, чтобы проверять разнообразие
COMMON_SUBSEQ_LENGTHS = (2,3)            # Длины подпоследовательностей операций, которые ищутся в траекториях оптимизации для выявления повторяющихся паттернов
TOP_COMMON_SUBSEQ = 10                   # Максимальное число наиболее частых подпоследовательностей, которое берётся для анализа и использования в стратегии
RECENT_GENERATIONS_FOR_DIVERSITY = 4     # Количество последних поколений, которые используются для оценки разнообразия популяции промптов

# Веса метрик для оценки промптов
METRIC_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.6,
    "safety": 0.2,
    "robustness": 0.2,
    "efficiency": 0.0,
    "f1": 0.0,
}