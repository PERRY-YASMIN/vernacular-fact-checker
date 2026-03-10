from pathlib import Path
import torch

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data"

# Data files
SAMPLE_POSTS_PATH = DATA_DIR / "sample_posts.jsonl"
VERIFIED_FACTS_PATH = DATA_DIR / "verified_facts.jsonl"

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CLAIM_DETECTOR_MODEL_NAME = "xlm-roberta-base"
CLAIM_DETECTOR_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "claim_detector"
CLAIM_DETECTOR_THRESHOLD = 0.5

# Verifier (Milestone 2): multilingual NLI for claim vs fact
# Premise = retrieved fact text, Hypothesis = extracted claim text
VERIFIER_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
VERIFIER_BATCH_SIZE = 8

# Retrieval
TOP_K_FACTS = 5
MIN_SIMILARITY = 0.4  # threshold for \"reasonable\" match
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_ALPHA = 0.7  # final = alpha * vector + (1-alpha) * bm25
RERANK_TOP_N = 20

# Preprocessing
ENABLE_TRANSLITERATION_FALLBACK = True


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"