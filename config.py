LOCAL_PARENTS_PER_ITERATION: int = 4     # Сколько лучших узлов использовать как «родителей»
MINI_BATCH_RATIO: float = 0.5            # Доля валидационных примеров для предварительного отбора
N_GRADIENTS: int = 4                     # Количество итераций семплирования ошибок для градиентов
ERRORS_PER_GRADIENT: int = 4             # Число примеров ошибок на один градиент
GRADIENTS_PER_ERROR: int = 1             # Число feedback-причин на один набор ошибок
STEPS_PER_GRADIENT: int = 1              # Число новых промптов на один feedback
MC_SAMPLES_PER_STEP: int = 2             # Monte Carlo synonym samples на каждый вариант
MAX_EXPANSION_FACTOR: int = 6            # Максимум кандидатов на beam member перед фильтрацией
REJECT_ON_ERRORS: bool = True            # Фильтровать кандидатов по ошибкам
LOCAL_TEMPERATURE: float = 0.1           # Температура для локальной оптимизации
LOCAL_BATCH_SIZE: int = 12               # Максимальное число failure-примеров в одном батче
MAX_GRADIENT_PAIRS: int = 2              # Макс. пар градиент-родитель за одну локальную итерацию
PRE_SCREEN_TOP_K: int = 4                # Сколько кандидатов полностью оценивать после предварительного отбора
TRAIN_FAILURE_SAMPLE_SIZE: int = 80      # Сколько примеров сэмплировать из train для поиска ошибок
DEFAULT_PRIORITY: float = 0.5            # Приоритет градиента по умолчанию

# Стратегия выбора wrong-exemplars для мета-промпта:
# "accumulative_most_frequent" — топ-K по накопленному счётчику провалов
# "current_most_frequent"      — топ-K по счётчику провалов среди инструкций мета-промпта
# "random"                     — случайная выборка, seed = current_generation (default)
# "constant"                   — фиксированная случайная выборка, seed = 0
EXEMPLAR_SELECTION_STRATEGY: str = "accumulative_most_frequent"
GLOBAL_CANDIDATES: int = 8               # Ширина глобального поиска
GLOBAL_HISTORY_WINDOW: int = 20          # Сколько узлов истории анализировать
GLOBAL_TEMPERATURE: float = 1.0          # Температура LLM для глобального оптимизатора
GLOBAL_MIN_IMPROVEMENT: float = 0.001    # Минимальный прирост для принятия global-кандидата
EXEMPLAR_COUNT: int = 5                  # Число QA-exemplars в мета-промпте
HISTORY_SCORE_THRESHOLD: float = 0.3     # Порог: 0.0 — все инструкции
MAX_INSTRUCTION_LENGTH: int = 700        # Макс. длина инструкции
STAGNATION_SIMILARITY_THRESHOLD: float = 0.7 # Порог сходства для диагностики застоя

MAX_GENERATIONS: int = 6                 # Максимальное число поколений
LOCAL_ITERATIONS_PER_GENERATION: int = 2 # Количество локальных итераций
GLOBAL_TRIGGER_INTERVAL: int = 2         # Каждые N поколений запускать глобальную оптимизацию
PATIENCE: int = 2                        # Поколений без улучшения до остановки
FORCE_GLOBAL_AFTER_STAGNATION: int = 2   # Форсировать глобальный шаг после N поколений без улучшения
MIN_IMPROVEMENT: float = 0.001           # Минимальное улучшение для продолжения
SIMILARITY_THRESHOLD: float = 0.80       # Порог схожести промптов

MAX_TOKENS: int = 3000                   # Максимальное число токенов
PROVIDER = "openai"                      # Провайдер LLM
API_KEY = ""                             # Ключ API провайдера
MODEL = "gpt-4o-mini"                    # Модель LLM

MAX_EXAMPLES_PER_NODE: int = 50          # Максимальное число примеров на узел
BATCH_EVAL_SIZE: int = 25                # Размер батча для групповых LLM-запросов при оценке
CORRECTNESS_TOKEN_F1_THRESHOLD: float = 0.5  # Порог token-F1 для определения правильности
STRICT_QA_TOKEN_F1_THRESHOLD: float = 0.8 # Per-example сигнал для QA split / failure mining
MIN_LIST_ITEM_LENGTH: int = 5            # Мин. длина элемента списка в ответе LLM
MIN_PROMPT_LENGTH: int = 20              # Минимальная длина валидного промпта
TOP_BEST_NODES: int = 5                  # Количество лучших узлов
MAX_DISTANCE_PAIRS: int = 10             # Максимальное число пар для попарных метрик
FALLBACK_ANALYSIS_LENGTH: int = 500      # Символов для fallback, если секция ERROR ANALYSIS пуста
DEFAULT_STAGNATION_WINDOW: int = 2       # Размер окна поколений для определения застоя
DIVERSITY_DISTANCE_THRESHOLD: float = 0.3 # Порог среднего расстояния для диверсификации
RECENT_GENERATIONS_FOR_DIVERSITY: int = 4 # Поколений для оценки разнообразия популяции
COMMON_WORDS_TOP_K: int = 20             # Наиболее частых слов при извлечении «общих фраз»
COMMON_WORD_MIN_FREQ: int = 3            # Минимальная частота слова для «общих фраз»

ENABLE_DIAGNOSTIC_LOGS: bool = True      # Включить диагностическое логирование

SQUAD_METRICS = [
    {"name": "exact_match",         "weight": 0.8, "stage": 1},
    {"name": "token_f1",            "weight": 0.2, "stage": 1},
]
GENERATION_METRICS = [
    {"name": "bertscore",  "weight": 0.5, "stage": 1},
    {"name": "rouge_l",    "weight": 0.3, "stage": 1},
    {"name": "meteor",     "weight": 0.2, "stage": 1},
]
GSM8K_METRICS = [
    {"name": "numeric_exact_match",  "weight": 1.0, "stage": 1}
]
XSUM_METRICS = [
    {"name": "rouge_l",   "weight": 0.5, "stage": 1},
    {"name": "bertscore", "weight": 0.35, "stage": 1},
    {"name": "meteor",    "weight": 0.15, "stage": 1},
]
METRICS_CONFIG = SQUAD_METRICS
