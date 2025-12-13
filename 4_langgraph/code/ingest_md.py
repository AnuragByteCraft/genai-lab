"""
Naive RAG - Implementation Example
Ingest md and mdx files - LangGraph documentation (cleaned)
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from traceback import print_exc
from langchain_text_splitters import RecursiveCharacterTextSplitter

import glob
import re
import os

# Paths
DATA_PATH = r"C:\home\ananth\research\packages\docs\src\oss"

device = "cuda"  # use "mps" for Mac

# CONTENT_ID = "langgraph_docs"
CONTENT_ID = "langchain_ai_docs"
DB_CHROMA_PATH = "vector_stores/db_chroma" + "_" + CONTENT_ID
EMBEDDINGS_MODEL = "thenlper/gte-large"


# ------------------------------------------------------
# CLEANING / NORMALIZATION FOR MDX FILES
# ------------------------------------------------------

def clean_mdx(text: str) -> str:
    """Convert MDX → clean Markdown by removing JSX noise and MDX components."""

    # --- Remove frontmatter ---
    text = re.sub(r"^---[\s\S]+?---", "", text, flags=re.MULTILINE)

    # --- Remove JSX components with bodies (Info, Tabs, Tip, Note, etc) ---
    text = re.sub(r"<[A-Za-z0-9_]+[^>]*>[\s\S]*?</[A-Za-z0-9_]+>", "", text)

    # --- Remove single tags (self-closing JSX like <Tip />) ---
    text = re.sub(r"<[^>]+/>", "", text)

    # --- Convert :::python → ```python ---
    text = re.sub(r":::python", "```python", text)
    text = re.sub(r":::js", "```javascript", text)
    text = re.sub(r":::", "```", text)

    # --- Remove leftover HTML tags ---
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


# ------------------------------------------------------
# LOAD MD + MDX FILES
# ------------------------------------------------------

def get_docs():
    """Load both .md and .mdx files with MDX cleaning applied."""

    filepaths = (
        glob.glob(os.path.join(DATA_PATH, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(DATA_PATH, "**", "*.mdx"), recursive=True)
    )

    docs = []
    for path in filepaths:
        try:
            raw = open(path, "r", encoding="utf-8").read()

            # Clean MDX files
            if path.endswith(".mdx"):
                cleaned = clean_mdx(raw)
            else:
                cleaned = raw

            # Create a LangChain Document manually
            from langchain_core.documents import Document
            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": path,
                    "source_path": path,
                    "filename": os.path.basename(path),
                    "filetype": "mdx" if path.endswith(".mdx") else "md"
                },
            )
            docs.append(doc)

        except Exception as e:
            print(f"Error reading {path}: {e}")

    return docs


# ------------------------------------------------------
# CHUNKING
# ------------------------------------------------------

def get_chunks(docs, chunk_size=2048, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


# ------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------

def get_embeddings_model(model_name=None, device=device):
    if model_name is None:
        model_name = EMBEDDINGS_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True}
    )


# ------------------------------------------------------
# VECTOR STORE
# ------------------------------------------------------

def create_vector_store(texts, embeddings, db_path, use_db="chroma"):
    try:
        if use_db == "chroma":
            db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
            db.persist()
            return True
        else:
            print("Unknown DB type.")
            return False

    except Exception:
        print_exc()
        print("Exception when creating data store:", db_path, use_db)
        return False


# ------------------------------------------------------
# MAIN INGEST FUNCTION
# ------------------------------------------------------

def ingest():
    docs = get_docs()

    print(f"Loaded {len(docs)} documents.")
    for d in docs[:5]:  # preview first few
        print(d.metadata)

    chunks = get_chunks(docs)
    print(f"Chunks created: {len(chunks)}")

    embeddings = get_embeddings_model()
    ok = create_vector_store(chunks, embeddings, DB_CHROMA_PATH)

    if ok:
        print("Vector Store Created!")


if __name__ == '__main__':
    ingest()

