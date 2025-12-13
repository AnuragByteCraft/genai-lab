"""
Naive RAG - Implementation Example
Ingest source code - python source code
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from traceback import print_exc

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# OPTIONAL: read_summaries gets the dict of file name versus summary for all python file names
# from core.summarizer.summarize_repo import read_summaries

# DATA_PATH = r"/Users/ananth/research/packages/nanochat"  # MAC path
DATA_PATH = r"C:\home\ananth\research\packages\nanochat"  # PC laptop path

# device = "mps"  # for MAC
device = "cuda"  # for PC and laptop

CONTENT_ID = "code"
DB_CHROMA_PATH = "vector_stores/db_chroma" + "_" + CONTENT_ID
EMBEDDINGS_MODEL = "thenlper/gte-large"
# EMBEDDINGS_MODEL = "nvidia/NV-Embed-v2"
# EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v4"  # more recent and multimodal


def get_docs():
    # Python files - Python Language Parser gives more granularity like functions_classes, etc
    py_loader = GenericLoader.from_filesystem(
        DATA_PATH,
        glob="**/*.py",
        parser=LanguageParser(language="python"),
    )

    # Load Rust files with simple TextLoader (no LanguageParser) - doesn't have AST granularity
    rs_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.rs",
        loader_cls=TextLoader,
    )

    # Load both
    python_docs = py_loader.load()
    rust_docs = rs_loader.load()

    # Merge results
    docs = python_docs + rust_docs

    # -------------------------------------- OPTIONAL STEPS ------------------------------------------------
    # # update the metadata with summary for each document object
    # summaries = read_summaries()
    #
    # # OPTIONAL: ingest the summaries either in the Vector db or keep it in the metadata for later use
    # for doc in docs:  # we will add summary to the metadata
    #     doc.metadata.update(summaries.get(doc.metadata["source"]))
    # --------------------------------------------------------------------------------------------------------

    return docs


def get_chunks(docs, chunk_size=1024, chunk_overlap=128):
    """
    Split documents by language, applying language-specific chunking strategies.
    :param docs: documents returned by loaders (Python + Rust files)
    :param chunk_size: size of each chunk in characters
    :param chunk_overlap: overlap between chunks in characters
    :return: list of chunked documents with metadata
    """
    # Create language-specific splitters
    splitters = {
        'python': RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ),
        'rust': RecursiveCharacterTextSplitter.from_language(
            language=Language.RUST,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ),
    }

    # Separate documents by file extension
    docs_by_language = {'python': [], 'rust': []}
    for doc in docs:
        source = doc.metadata.get('source', '')
        if source.endswith('.py'):
            docs_by_language['python'].append(doc)
        elif source.endswith('.rs'):
            docs_by_language['rust'].append(doc)

    # Process each language batch
    all_chunks = []
    language_stats = {}

    for lang, batch_docs in docs_by_language.items():
        if not batch_docs:
            continue

        splitter = splitters[lang]
        chunks = splitter.split_documents(batch_docs)

        # Add metadata: preserve filename, chunk index, and language
        for i, chunk in enumerate(chunks):
            if 'source' in chunk.metadata:
                chunk.metadata['chunk_id'] = f"{chunk.metadata['source']}_{i}"
            chunk.metadata['language'] = lang

        all_chunks.extend(chunks)
        language_stats[lang] = len(chunks)

    # Debug: Show chunk distribution by language
    print(f"\n{'='*60}")
    print(f"Chunk Distribution by Language:")
    print(f"{'='*60}")
    for lang, count in language_stats.items():
        print(f"  {lang.upper()}: {count} chunks")

    # Debug: Show overall chunk statistics
    chunk_sizes = [len(t.page_content.split()) for t in all_chunks]
    print(f"\nOverall Statistics:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Avg chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} tokens")
    print(f"  Min/Max: {min(chunk_sizes)} / {max(chunk_sizes)}")

    # Debug: Show per-language statistics
    print(f"\nPer-Language Statistics:")
    for lang in ['python', 'rust']:
        lang_chunks = [t for t in all_chunks if t.metadata.get('language') == lang]
        if lang_chunks:
            lang_sizes = [len(t.page_content.split()) for t in lang_chunks]
            print(f"  {lang.upper()}:")
            print(f"    Avg chunk size: {sum(lang_sizes) / len(lang_sizes):.0f} tokens")
            print(f"    Min/Max: {min(lang_sizes)} / {max(lang_sizes)}")

    print(f"{'='*60}\n")

    return all_chunks


def get_embeddings_model(model_name=None, device=device):
    if model_name is None:
        model_name = EMBEDDINGS_MODEL
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs={"device": device, "trust_remote_code": True})
    return embeddings_model


def create_vector_store(texts, embeddings, db_path, use_db="chroma"):
    """
    Given the chunks, their embeddings and path to save the db, save and persist the data in the data store
    :param texts: chunks for which we are constructing the data store
    :param embeddings: vector embeddings for given chunks
    :param db_path: storage path
    :param use_db: type of data store to use
    :return: None
    """
    flag = True
    try:
        if use_db == "chroma":
            db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
        else:
            print("Unknown db type, exiting!")
            db = None
            import sys
            sys.exit(-1)
        db.persist()
    except:
        flag = False
        print_exc()
        print("Exception when creating data store: ", db_path, use_db)
    return flag


def ingest():
    """
    Ingest PDF files from the given data source.
    :return:
    """

    # 1. Load the data from the data source
    docs = get_docs()
    for doc in docs:
        print(doc.metadata)
    print(len(docs))
    texts = get_chunks(docs)

    print(f"Number of documents from get_docs: {len(docs)}, Number of chunks from get_chunks: {len(texts)}")

    embs_model = get_embeddings_model(device=device)
    flag = create_vector_store(texts, embs_model, DB_CHROMA_PATH, use_db="chroma")
    if flag:
        print("Vector Store Created!")


if __name__ == '__main__':
    ingest()
