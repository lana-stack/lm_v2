import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DEVICE
# ============================================================

if torch.xpu.is_available():
    DEVICE = "xpu"
    print("Intel XPU detected:", torch.xpu.get_device_name(0))
else:
    DEVICE = "cpu"
    print("XPU not available. Using CPU.")

# ============================================================
# MODEL
# ============================================================

N_EMBD = 512
N_HEAD = 8
N_LAYER = 8
BLOCK_SIZE = 512

# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 8

LEARNING_RATE = 3e-4

TRAIN_STEPS_PER_SESSION = 1000

VALIDATION_INTERVAL = 1000

# ============================================================
# CONTINUOUS LEARNING
# ============================================================

TRAIN_INTERVAL_SECONDS = 0

EVAL_EVERY_SESSIONS = 1

SAVE_EVERY_SESSION = True

# ============================================================
# GENERATION
# ============================================================

TEMPERATURE = 0.8
GENERATE_TOKENS = 1000

# ============================================================
# INTERNET
# ============================================================

COLLECT_EVERY_SESSIONS = 1

WIKIPEDIA_ARTICLES_PER_COLLECTION = 5

# Начальные темы.
WIKIPEDIA_TOPICS = [
    "Physics",
    "Biology",
    "Chemistry",
    "Mathematics",
    "Computer science",
    "History",
    "Philosophy",
    "Psychology",
    "Artificial intelligence",
    "Neuroscience",
]

WIKIPEDIA_TOPICS_RU = [
    "Физика",
    "Биология",
    "Химия",
    "Математика",
    "Информатика",
    "История",
    "Философия",
    "Психология",
    "Искусственный интеллект",
    "Нейронаука",
]

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = os.path.join(
    BASE_DIR,
    "input.txt"
)

CORPUS_DIR = os.path.join(
    BASE_DIR,
    "corpus"
)

WIKI_DIR = os.path.join(
    CORPUS_DIR,
    "wikipedia"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mind_model.pth"
)

STATE_PATH = os.path.join(
    BASE_DIR,
    "mind_state.json"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

SAMPLES_DIR = os.path.join(
    BASE_DIR,
    "samples"
)

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(WIKI_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
