import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/faqs.csv")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    df["question"].tolist()
)

with open(
    "models/faq_model.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "questions": df["question"].tolist(),
            "answers": df["answer"].tolist(),
            "embeddings": embeddings
        },
        f
    )
