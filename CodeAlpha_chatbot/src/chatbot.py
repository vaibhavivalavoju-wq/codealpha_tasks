import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

with open(
    "models/faq_model.pkl",
    "rb"
) as f:

    data = pickle.load(f)

faq_embeddings = data["embeddings"]
answers = data["answers"]
questions = data["questions"]


def get_response(user_query):

    query_embedding = model.encode(
        [user_query]
    )

    similarities = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    best_index = np.argmax(
        similarities
    )

    confidence = similarities[
        best_index
    ]

    if confidence < 0.50:
        return {
            "answer":
            "Sorry, I couldn't find a relevant answer.",
            "confidence":
            round(confidence, 2),
            "suggestions": []
        }

    top_indices = similarities.argsort()[-3:][::-1]

    suggestions = [
        questions[i]
        for i in top_indices
    ]

    return {
        "answer":
        answers[best_index],
        "confidence":
        round(confidence, 2),
        "suggestions":
        suggestions
    }
