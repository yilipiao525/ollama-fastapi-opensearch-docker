import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import requests
from langchain_core.documents import Document
import pdfplumber
import time
# Configure logging to display information and error messages
logging.basicConfig(level=logging.INFO)

# Constants for file paths and model configurations
DOC_PATH = "./data/"  # Path to the PDF document
MODEL_NAME = "gemma2:2b"      # Language model name
EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model name
VECTOR_STORE_NAME = "elf-test-index"       # Name for the vector database collection

def ingest_pdfs(directory_path):
    """Load all PDF documents from a directory using UnstructuredPDFLoader."""
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                with pdfplumber.open(file_path) as pdf:
                    logging.info(f"Processing PDF {filename} with {len(pdf.pages)} pages")
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        documents.append(Document(page_content=text, metadata={"source": filename}))
                        logging.info(f"Successfully extracted {len(text)} characters from {filename}")
                    else:
                        raise ValueError("No text extracted")
                        
            except Exception as e:
                logging.warning(f"pdfplumber failed for {filename}, trying UnstructuredPDFLoader: {str(e)}")
                try:
                    # Fallback to UnstructuredPDFLoader
                    loader = UnstructuredPDFLoader(
                        file_path=file_path,
                        strategy="fast",  # or "accurate" if needed
                        mode="elements"
                    )
                    data = loader.load()
                    if data:
                        documents.extend(data)
                        logging.info(f"Successfully loaded {filename} using UnstructuredPDFLoader")
                    else:
                        logging.error(f"No text extracted from {filename} using UnstructuredPDFLoader")
                        
                except Exception as e2:
                    logging.error(f"Both PDF extraction methods failed for {filename}: {str(e2)}")
                    continue

    if not documents:
        logging.error(f"No text could be extracted from any PDFs in directory: {directory_path}")
        raise ValueError("No text could be extracted from PDFs")
        
    return documents
# def ingest_pdfs(directory_path):
#     """Load all PDF documents from a directory using UnstructuredPDFLoader."""
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(directory_path, filename)
#             loader = UnstructuredPDFLoader(file_path=file_path)
#             data = loader.load()
#             documents.extend(data)
#             logging.info(f"PDF loaded successfully from {file_path}.")
#     if not documents:
#         logging.error(f"No PDF files found in directory: {directory_path}")
#     return documents


def split_documents(documents):
    """Split documents into smaller chunks for processing."""
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from the provided document chunks using OpenSearch."""
    try:
        # Test OpenSearch connection
        response = requests.get("http://opensearch:9200/_cluster/health")
        logging.info(f"OpenSearch health check: {response.json()}")

        # Test Ollama connection
        ollama_response = requests.get("http://ollama:11434/api/version")
        logging.info(f"Ollama health check: {ollama_response.json()}")

        # Pull embedding model using HTTP API instead of ollama client
        try:
            logging.info(f"Pulling embedding model: {EMBEDDING_MODEL}")
            pull_response = requests.post(
                "http://ollama:11434/api/pull",
                json={"name": EMBEDDING_MODEL},
                timeout=60
            )
            pull_response.raise_for_status()
            logging.info("Successfully pulled embedding model")
        except Exception as e:
            logging.error(f"Failed to pull embedding model: {str(e)}")
            raise

        # Initialize embeddings with only required parameters
        try:
            embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://ollama:11434/api/embed"
            )
            logging.info("Initialized embeddings object")
            
            # Test embeddings with error details
            try:
                test_text = "Test embedding generation"
                test_embedding = embeddings.embed_query(test_text)
                logging.info(f"Successfully generated test embedding. Vector length: {len(test_embedding)}")
            except Exception as e:
                logging.error(f"Failed to generate test embedding: {str(e)}")
                raise
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {str(e)}")
            raise

        # Create vector database
        vector_db = OpenSearchVectorSearch(
            index_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            opensearch_url="http://opensearch:9200",
            http_auth=("admin", "admin"),
            use_ssl=False
        )
        
        vector_db.add_documents(chunks)
        logging.info(f"Vector database '{VECTOR_STORE_NAME}' created/updated in OpenSearch.")
        return vector_db

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error creating vector database: {str(e)}")
        raise


def create_retriever(vector_db, llm):
    """Create a multi-query retriever using the vector database and language model."""
    # Define a prompt template to generate multiple versions of the user's question
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    # Create the retriever using the MultiQueryRetriever class
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, prompt):
    """Create a simpler chain using direct Ollama API calls"""
    try:
        # Define the base template
        template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
        
        def generate_response(context, question):
            """Inner function to call Ollama API directly"""
            prompt = template.format(context=context, question=question)
            
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5,
                    "top_k": 40,
                    "top_p": 0.9
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
            
        return generate_response

    except Exception as e:
        logging.error(f"Error creating chain: {str(e)}")
        raise


def connect_to_existing_vector_db():
    """Connect to an existing OpenSearch vector database."""
    try:
        # Create a custom embedding class that uses direct HTTP requests
        class CustomEmbeddings:
            def embed_query(self, text):
                try:
                    response = requests.post(
                        "http://ollama:11434/api/embeddings",
                        json={
                            "model": EMBEDDING_MODEL,
                            "prompt": text,
                            "options": {
                                "temperature": 0.0
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    logging.info(f"Raw embedding response: {result}")
                    
                    # Handle different response formats
                    if "embedding" in result:
                        embeddings = result["embedding"]
                    elif "embeddings" in result:
                        embeddings = result["embeddings"]
                    else:
                        raise ValueError(f"No embeddings in response: {result}")
                        
                    # Convert single float to list if needed
                    if isinstance(embeddings, (float, int)):
                        embeddings = [embeddings]
                    elif isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], (float, int)):
                        embeddings = embeddings
                    else:
                        raise ValueError(f"Unexpected embedding format: {type(embeddings)}")
                        
                    logging.info(f"Generated embedding vector of length: {len(embeddings)}")
                    return embeddings

                except Exception as e:
                    logging.error(f"Embedding error for text '{text[:100]}...': {str(e)}")
                    raise

            def embed_documents(self, texts):
                all_embeddings = []
                for text in texts:
                    embedding = self.embed_query(text)
                    if embedding:  # Only add non-empty embeddings
                        all_embeddings.append(embedding)
                return all_embeddings

        # Create the vector store with the custom embedding class
        embeddings = CustomEmbeddings()
        
        # Test the embeddings first
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            raise ValueError("Test embedding failed - empty vector received")
        logging.info(f"Test embedding successful, vector length: {len(test_embedding)}")

        vector_db = OpenSearchVectorSearch(
            index_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            opensearch_url="http://opensearch:9200",
            http_auth=("admin", "admin"),
            use_ssl=False,
            engine="nmslib",  # Specify the engine
            space_type="cosinesimil"  # Specify similarity metric
        )

        logging.info(f"Connected to existing vector database '{VECTOR_STORE_NAME}'.")
        return vector_db

    except Exception as e:
        logging.error(f"Error connecting to vector database: {str(e)}")
        raise

def process_and_create_vector_db(directory_path):
    """
    Load and process PDF documents from the specified directory,
    split them into chunks, and create a vector database.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        OpenSearchVectorSearch: The created vector database instance.
    """
    try:
        # Load all PDF documents from the directory
        data = ingest_pdfs(directory_path)
        if not data:
            logging.error("No valid PDFs found to process. Exiting the function.")
            return None

        # Split the documents into smaller chunks
        chunks = split_documents(data)
        logging.info(f"Document split into {len(chunks)} chunks.")

        # Create the vector database from the chunks
        vector_db = create_vector_db(chunks)
        logging.info("Vector database created successfully.")
        return vector_db

    except Exception as e:
        logging.error(f"Error processing and creating vector database: {str(e)}")
        return None
