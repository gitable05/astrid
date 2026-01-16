from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
CHROMA_PATH = "chroma"
REFERENCES_PATH = "sample"

vector_store = Chroma(collection_name="references",
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings)

documents = PyPDFDirectoryLoader(REFERENCES_PATH).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                chunk_overlap=80,
                                                length_function=len,
                                                is_separator_regex=False)
chunks = text_splitter.split_documents(documents)
vector_store.add_documents(chunks)


