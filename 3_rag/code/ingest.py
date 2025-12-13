"""
Naive RAG - Implementation Example

"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from traceback import print_exc


DATA_PATH = r"C:\home\ananth\research\my_projects\agentic_ai_dec2025_2026\core\rag_agents\dataset"
DB_CHROMA_PATH = "vector_stores/db_chroma_annual_reports"
EMBEDDINGS_MODEL = "thenlper/gte-large"


def get_docs():
    """
    loads the documents from the given source where each doc has page_content and metadata.
    It is possible to add metadata by updating the doc.metadata which is a dict()
    :return:
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)
    docs = loader.load()
    return docs


def get_chunks(docs, chunk_size=512, chunk_overlap=50):
    """
    Given docs obtained by using LangChain loader, split these to chunks and return them
    :param docs: documents returned by loader, that could be ArxivLoader or local directory loader
    :param chunk_size: size of chunk to be set according to the application
    :param chunk_overlap: overlap window size between consecutive chunks
    :return: chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts  # chunked docs


def get_embeddings_model(model_name=None, device="cuda"):
    if model_name is None:
        model_name = EMBEDDINGS_MODEL
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={"device": device})
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
    :param source: Can be "arxiv" to load from arxiv web site, "local" for DirectoryLoader
    :return:
    """

    # 1. Load the data from the data source
    docs = get_docs()
    # print(docs[0])
    # docs = get_docs(source="arxiv")

    # 2. Chunk the documents
    texts = get_chunks(docs)
    print(len(docs), len(texts))

    # 3. get the embedding model
    embs_model = get_embeddings_model(device="cuda")

    # 4. Use the embedding model to vectorize and save it in db
    flag = create_vector_store(texts, embs_model, DB_CHROMA_PATH, use_db="chroma")
    if flag:
        print("Vector Store Created!")


def get_retriever():
    """
    after texts are ingested in vectordb, get it as a retriever
    :return:
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                       model_kwargs={'device': 'cuda'})
    vectordb = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 8})


if __name__ == '__main__':
    ingest()


