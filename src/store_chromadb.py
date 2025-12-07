import os
from pathlib import Path
import uuid
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data" / "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.replace(" ", "_")

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def meta_str(x):
    if x is None or pd.isna(x):
        return ""
    return str(x)

df["title"] = df["product_name"].apply(safe_str)
df["brand"] = df["brand_name"].apply(safe_str)
df["category"] = df["category"].apply(safe_str)

def parse_price(x):
    try:
        return float(str(x).replace("$", "").strip())
    except:
        return None

df["price"] = df["selling_price"].apply(parse_price)
df.loc[df["price"].isna(), "price"] = df["list_price"].apply(parse_price)

feature_cols = [
    "about_product",
    "product_specification",
    "technical_details",
    "product_details"
]

df["features"] = df[feature_cols].apply(
    lambda row: " ".join(safe_str(x) for x in row if safe_str(x)), axis=1
)

df["review_snippets"] = df["product_description"].apply(safe_str)
df["ingredients"] = df["ingredients"].apply(safe_str)
df["doc_id"] = df["uniq_id"].apply(safe_str)

def build_embedding_text(row):
    return "\n".join([
        f"Title: {row['title']}",
        f"Features: {row['features']}",
        f"Description: {row['review_snippets']}",
        f"Brand: {row['brand']}",
        f"Category: {row['category']}",
        f"Ingredients: {row['ingredients']}",
    ])

df["embedding_text"] = df.apply(build_embedding_text, axis=1)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CHROMA_PATH = project_root / "data" / "chroma_amazon_clean"
os.makedirs(CHROMA_PATH, exist_ok=True)

client = PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="amazon_products",
    metadata={"hnsw:space": "cosine"}
)

documents = df["embedding_text"].tolist()
metas = []
ids = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building metadata"):
    meta = {
        "doc_id": meta_str(row["doc_id"]),
        "title": meta_str(row["title"]),
        "brand": meta_str(row["brand"]),
        "category": meta_str(row["category"]),
        "price": meta_str(row["price"]),
        "rating": meta_str(""),
        "ingredients": meta_str(row["ingredients"]),
    }
    metas.append(meta)
    ids.append(str(uuid.uuid4()))

BATCH_SIZE = 64

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding & inserting"):
    batch_docs = documents[i : i + BATCH_SIZE]
    embeddings = embedder.encode(batch_docs, convert_to_numpy=True, show_progress_bar=False)
    batch_metas = metas[i : i + BATCH_SIZE]
    batch_ids = ids[i : i + BATCH_SIZE]

    collection.add(
        embeddings=embeddings,
        documents=batch_docs,
        metadatas=batch_metas,
        ids=batch_ids,
    )

print("[Chroma DB]")
print(f"Chroma DB saved at: {CHROMA_PATH}")
