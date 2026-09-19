import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DATASET_CSV_PATH = "bugs_dataset.csv"  # Name of your downloaded Kaggle CSV file
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "aibafa_vector_store"
BATCH_SIZE = 256  # Adjust based on your system memory

def seed_database():
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"❌ Error: CSV file '{DATASET_CSV_PATH}' not found in the project directory.")
        print("Please place your downloaded Kaggle CSV file in this folder and update DATASET_CSV_PATH.")
        return

    print(f"1. Loading dataset from '{DATASET_CSV_PATH}'...")
    df = pd.read_csv(DATASET_CSV_PATH)
    
    # Normalize column headers to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"   Found columns: {list(df.columns)}")

    # Map dataset columns (Auto-detect common Kaggle column names)
    summary_col = next((c for c in ['short_description', 'summary', 'title', 'bug_title'] if c in df.columns), None)
    desc_col = next((c for c in ['long_description', 'description', 'bug_report', 'details'] if c in df.columns), None)
    comp_col = next((c for c in ['component_name', 'component', 'category', 'product_name'] if c in df.columns), None)

    print(f"   Using summary column: '{summary_col}' | description column: '{desc_col}'")

    bug_texts = []
    ids = []
    metadatas = []

    for idx, row in df.iterrows():
        summary = str(row[summary_col]) if summary_col and pd.notna(row[summary_col]) else ""
        description = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else ""
        category = str(row[comp_col]) if comp_col and pd.notna(row[comp_col]) else "Public Bug Repository"
        
        text_content = f"{summary} | {description}".strip(" |")
        
        # Filter out empty or extremely short rows
        if text_content and len(text_content) > 10:
            bug_id = f"public_bug_{idx}"
            bug_texts.append(text_content)
            ids.append(bug_id)
            metadatas.append({
                "severity": "MEDIUM", 
                "category": category[:50]  # Truncate string for storage cleanliness
            })

    total_bugs = len(bug_texts)
    print(f"2. Extracted {total_bugs} valid bug reports.")

    # Load local embedding model & ChromaDB Client
    print("3. Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"4. Generating embeddings and indexing into ChromaDB in batches of {BATCH_SIZE}...")
    for i in range(0, total_bugs, BATCH_SIZE):
        batch_texts = bug_texts[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]

        # Generate vectors
        embeddings = embedding_model.encode(
            batch_texts, 
            batch_size=128, 
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Upsert into vector database
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas
        )
        print(f"   Indexed [{i + len(batch_texts)} / {total_bugs}] documents...")

    print(f"🎉 Success! Vector Store now contains {collection.count()} public defect documents.")

if __name__ == "__main__":
    seed_database()