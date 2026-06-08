import pickle
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Logger setup (optional but pro)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("FAQ-BOT")


# -----------------------------
# Load model / data
# -----------------------------
def load_pickle(file_path):
    """Load trained FAQ model"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Model loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# -----------------------------
# Save model / data
# -----------------------------
def save_pickle(data, file_path):
    """Save trained FAQ model"""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


# -----------------------------
# Cosine similarity
# -----------------------------
def get_similarity(query_vector, faq_vectors):
    """Return similarity scores"""
    return cosine_similarity(query_vector, faq_vectors)[0]


# -----------------------------
# Top K results
# -----------------------------
def get_top_k_indices(similarities, k=3):
    """Get top K most similar FAQ indices"""
    return np.argsort(similarities)[-k:][::-1]


# -----------------------------
# Confidence check
# -----------------------------
def is_confident(score, threshold=0.5):
    """Check if model is confident enough"""
    return score >= threshold
