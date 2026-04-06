"""
Configuration for the prompt optimization system.

Contains all hyperparameters for local and global optimization,
LLM provider settings, metric thresholds, and metric set
configurations for various tasks (SQuAD, generation, GSM8K, XSum).
"""

import os

# ── Local optimization parameters ────────────────────────────────────────────

LOCAL_PARENTS_PER_ITERATION: int = 4     # Number of top nodes to use as parents each iteration
MINI_BATCH_RATIO: float = 0.5            # Fraction of validation examples used for pre-screening
N_GRADIENTS: int = 4                     # Number of error-sampling iterations for gradient generation
ERRORS_PER_GRADIENT: int = 4             # Number of failure examples per gradient
GRADIENTS_PER_ERROR: int = 1             # Number of feedback reasons per error batch
STEPS_PER_GRADIENT: int = 1              # Number of new prompts generated per feedback
MC_SAMPLES_PER_STEP: int = 2             # Monte Carlo synonym samples per candidate
MAX_EXPANSION_FACTOR: int = 6            # Max candidates per beam member before filtering
REJECT_ON_ERRORS: bool = True            # Whether to filter candidates that still produce errors
LOCAL_TEMPERATURE: float = 0.1           # LLM temperature for local optimization
LOCAL_BATCH_SIZE: int = 12               # Max number of failure examples in a single batch
MAX_GRADIENT_PAIRS: int = 2              # Max gradient-parent pairs per local iteration
PRE_SCREEN_TOP_K: int = 4                # Number of candidates to fully evaluate after pre-screening
TRAIN_FAILURE_SAMPLE_SIZE: int = 40      # Number of examples to sample from train set for failure mining
DEFAULT_PRIORITY: float = 0.5            # Default gradient priority

# Strategy for selecting wrong-exemplars for the meta-prompt:
# "accumulative_most_frequent" — top-K by accumulated failure counter across all history
# "current_most_frequent"      — top-K by failure count for the current best prompt
# "random"                     — random sample, seed = current_generation
# "constant"                   — fixed random sample, seed = 0
EXEMPLAR_SELECTION_STRATEGY: str = "current_most_frequent"

# ── Global optimization parameters ──────────────────────────────────────────

GLOBAL_CANDIDATES: int = 16              # Number of candidates generated in the global search step
GLOBAL_HISTORY_WINDOW: int = 10          # Number of history nodes to include in the meta-prompt
GLOBAL_TEMPERATURE: float = 0.75         # LLM temperature for global optimization (higher = more exploration)
GLOBAL_MIN_IMPROVEMENT: float = 0.001    # Minimum score gain required to accept a global candidate
EXEMPLAR_COUNT: int = 3                  # Number of QA exemplars to include in the meta-prompt
HISTORY_SCORE_THRESHOLD: float = 0.5     # Only nodes with score >= this threshold are included in meta-prompt
MAX_INSTRUCTION_LENGTH: int = 700        # Maximum allowed length (characters) for a prompt
STAGNATION_SIMILARITY_THRESHOLD: float = 0.65 # Similarity threshold for stagnation detection

# ── Optimization loop control ────────────────────────────────────────────────

MAX_GENERATIONS: int = 6                 # Maximum number of optimization generations
LOCAL_ITERATIONS_PER_GENERATION: int = 2 # Number of local optimization iterations per generation
GLOBAL_TRIGGER_INTERVAL: int = 2         # Run global optimization every N generations
PATIENCE: int = 3                        # Generations without improvement before early stopping
FORCE_GLOBAL_AFTER_STAGNATION: int = 1   # Force a global step after N generations without improvement
MIN_IMPROVEMENT: float = 0.001           # Minimum score improvement to count as progress
SIMILARITY_THRESHOLD: float = 0.80       # Cosine similarity threshold for duplicate prompt detection

# ── LLM provider settings ───────────────────────────────────────────────────

MAX_TOKENS: int = 3000                   # Maximum number of tokens in LLM response
EVAL_TEMPERATURE: float = 0.1            # LLM temperature during prompt evaluation (deterministic)
EVAL_SEED: int = 42                      # Base random seed for example sampling during evaluation
PROVIDER: str = "openai"                 # LLM provider: "openai", "gemini", or "local"
API_KEY = os.environ.get("OPENAI_API_KEY", "")  # API key — set via OPENAI_API_KEY environment variable
MODEL = "gpt-4o-mini"                    # LLM model name

# ── Evaluation and metrics settings ─────────────────────────────────────────

MAX_EXAMPLES_PER_NODE: int = 50          # Maximum number of examples evaluated per prompt node
BATCH_EVAL_SIZE: int = 25                # Batch size for grouped LLM calls during evaluation
CORRECTNESS_TOKEN_F1_THRESHOLD: float = 0.5  # Token-F1 threshold for marking an answer as correct
STRICT_QA_TOKEN_F1_THRESHOLD: float = 0.8 # Per-example threshold for QA split / failure mining
MIN_LIST_ITEM_LENGTH: int = 5            # Minimum length of a list item in an LLM response
MIN_PROMPT_LENGTH: int = 20              # Minimum character length for a valid prompt
TOP_BEST_NODES: int = 5                  # Number of top-scoring nodes to track
MAX_DISTANCE_PAIRS: int = 10             # Maximum number of pairs for pairwise distance metrics
FALLBACK_ANALYSIS_LENGTH: int = 500      # Characters to use as fallback if ERROR ANALYSIS section is empty
DEFAULT_STAGNATION_WINDOW: int = 2       # Generation window size for stagnation detection
DIVERSITY_DISTANCE_THRESHOLD: float = 0.3 # Mean edit-distance threshold for diversity trigger
RECENT_GENERATIONS_FOR_DIVERSITY: int = 4 # Number of recent generations to assess population diversity
COMMON_WORDS_TOP_K: int = 20             # Top-K most frequent words when extracting common phrases
COMMON_WORD_MIN_FREQ: int = 3            # Minimum word frequency to be considered a common phrase

ENABLE_DIAGNOSTIC_LOGS: bool = True      # Enable verbose diagnostic logging

# ── Metric configurations for supported tasks ────────────────────────────────

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
