"""
Naive RAG - Implementation Example
Ingest md and mdx files - LangGraph documentation (cleaned)
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from traceback import print_exc
from langchain_community.document_loaders import TextLoader
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


# """
# Naive RAG - Implementation Example
# Ingest md files - LangGraph documentation
# """
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from traceback import print_exc
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
#
# # DATA_PATH = r"/Users/ananth/research/packages/nanochat"  # MAC path
# # DATA_PATH = r"C:\home\ananth\research\packages\langgraph"
#
# DATA_PATH = r"C:\home\ananth\research\packages\docs\src\oss"
#
# # device = "mps"  # for MAC
# device = "cuda"  # for PC and laptop
#
# CONTENT_ID = "langgraph_docs"
# DB_CHROMA_PATH = "vector_stores/db_chroma" + "_" + CONTENT_ID
# EMBEDDINGS_MODEL = "thenlper/gte-large"
# # EMBEDDINGS_MODEL = "nvidia/NV-Embed-v2"
# # EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v4"  # more recent and multimodal
#
#
# def get_docs():
#     loader = DirectoryLoader(
#         DATA_PATH,
#         glob="**/*.md",
#         loader_cls=TextLoader,
#         loader_kwargs={"encoding": "utf-8"}
#     )
#
#     docs = loader.load()
#
#     for d in docs:
#         d.metadata["source_path"] = d.metadata["source"]
#         d.metadata["filename"] = d.metadata["source"].split("\\")[-1]  # for Windows paths
#
#     docs = loader.load()
#
#     return docs
#
#
# def get_chunks(docs, chunk_size=2048, chunk_overlap=200):
#     """
#     Given docs obtained by using LangChain loader, split these to chunks and return them
#     :param docs: documents returned by loader, that could be ArxivLoader or local directory loader
#     :param chunk_size: size of chunk to be set according to the application
#     :param chunk_overlap: overlap window size between consecutive chunks
#     :return: chunks
#     """
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     texts = text_splitter.split_documents(docs)
#     return texts  # chunked docs
#
#
# def get_embeddings_model(model_name=None, device=device):
#     if model_name is None:
#         model_name = EMBEDDINGS_MODEL
#     embeddings_model = HuggingFaceEmbeddings(model_name=model_name,
#                                        model_kwargs={"device": device, "trust_remote_code": True})
#     return embeddings_model
#
#
# def create_vector_store(texts, embeddings, db_path, use_db="chroma"):
#     """
#     Given the chunks, their embeddings and path to save the db, save and persist the data in the data store
#     :param texts: chunks for which we are constructing the data store
#     :param embeddings: vector embeddings for given chunks
#     :param db_path: storage path
#     :param use_db: type of data store to use
#     :return: None
#     """
#     flag = True
#     try:
#         if use_db == "chroma":
#             db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
#         else:
#             print("Unknown db type, exiting!")
#             db = None
#             import sys
#             sys.exit(-1)
#         db.persist()
#     except:
#         flag = False
#         print_exc()
#         print("Exception when creating data store: ", db_path, use_db)
#     return flag
#
#
# def ingest():
#     """
#     Ingest PDF files from the given data source.
#     :return:
#     """
#
#     # 1. Load the data from the data source
#     docs = get_docs()
#     for doc in docs:
#         print(doc.metadata)
#     print(len(docs))
#     texts = get_chunks(docs)
#
#     print(f"Number of documents from get_docs: {len(docs)}, Number of chunks from get_chunks: {len(texts)}")
#
#     embs_model = get_embeddings_model(device=device)
#     flag = create_vector_store(texts, embs_model, DB_CHROMA_PATH, use_db="chroma")
#     if flag:
#         print("Vector Store Created!")
#
#
# if __name__ == '__main__':
#     ingest()
