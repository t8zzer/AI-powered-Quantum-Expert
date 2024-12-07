# AI-powered-Quantum-Expert

#Note: Be aware to have the latest version of python installed! (If you want to run this file locally, please make sure to use a python version less than or equal to 12)
## API Keys: You need to obtain API keys for OpenAI and Nomic:
### get API Key for OpenAI: https://platform.openai.com/api-keys
### get API key for Nomic: https://atlas.nomic.ai/data/<username>/org/keys

## 1. Installations (required python installations)
    #pip install python-dotenv 
    #pip install llama-index openai
    #pip install llama-index-vector-stores-faiss faiss-cpu
    #pip install faiss-cpu
    #pip install -U llama-index llama-index-embeddings-nomic
    #pip install nest_asyncio

    python-dotenv: to load environment variables (e.g. API-keys) from an .env file
    llama-index, openai: to interact with llamaindex and OpenAI: Handle document indexing and querying using OpenAI models
    faiss-cpu: to use the efficient FAISS-library for vector indicies
    nest_asyncio: to execute asynchronous tasks in jupyter notebooks

## 2. Imports
    dotenv: To load environment variables from a .env file (e.g., the API keys).
    faiss: Used for handling the FAISS library (used to efficiently store and query high-dimensional vectors).
    nest_asyncio: Helps running asynchronous code in environments like Jupyter notebooks.
    llama_index: A module for indexing, querying, and working with document embeddings (uses OpenAI and Nomic models)

## 3. Load API Keys
    load_dotenv() - loads environment variables from the .env file
    os.getenv() - retrieves the values of OPENAI_API_KEY and NOMIC_API_KEY from the environment variables

## 4. Initializing the Embedding Model and Language Model
    NomicEmbedding: The Nomic embedding model (nomic-embed-text-v1.5) is initialized with a specific API key and dimensionality (256). This model is used to transform text into vectors.
    OpenAI: The gpt-3.5-turbo model is chosen to process language-based tasks (like answering questions based on context).

## 5. Changing Global Settings
    Settings.llm and Settings.embed_model set the LLM and embedding model globally for LlamaIndex.

## 6. Logging Setup
    logging.basicConfig configures logging to print all INFO level messages to the console. 
    The logging level is set to INFO. The logging output is printed to the standard output (stdout).
    An instance of logger is created, which can be used to log messages throughout the script.

## 7. Creating Index Storage
    Checks if the storage directory (./storage) already exists. If not, a new index will be created.
    If the index already exists (i.e., the ./storage directory exists), the code will load the existing index from storage.
    Note: If the storage already exists, you can't load more documents into the data folder or you must delete the existing storage.

## 8. Loading Documents and Creating the Index
    The SimpleDirectoryReader reads documents from the given directory (data).
    The chunk size specifies how large each chunk of text will be (256 characters).
    The chunk overlap defines how many tokens (words or characters) should overlap between adjacent chunks to preserve context when splitting text.

## 9. Creating the Vector Store
    faiss.IndexFlatL2(d) - creates a flat L2 (Euclidean) index for high-dimensional vectors with d = 256 (the dimensionality of the Nomic embeddings).
    FaissVectorStore - vector store built using the FAISS index to store and query embeddings efficiently.

## 10. Creating and Persisting the Index
    VectorStoreIndex.from_documents - creates an index from the documents using the vector store created earlier.
    index.storage_context.persist - saves the index to the PERSIST_DIR directory for later use

## 11. Querying the Index
    First, a custom prompt template is defined, where the model acts as a tutor named "TutorGPT-2.0". 
    The model is expected to answer student questions based on the context (documents) provided to it.
    Second, the PromptTemplate object is created using the custom template. It will format questions and context to feed to the model.

    query_engine=
    index.as_query_engine - converts the index into a query engine that can be used to search the index for relevant documents
        similarity_top_k=4 - the engine will return the top 4 most similar documents when a query is made
        text_qa_template=qa_template - the query engine uses the custom prompt template to format and process the query

## 12. Performing the Query
    The query engine executes a query ("<Insert your question here>?") on the index.
    Last, the the response from the query is printed, which should be an answer based on the indexed documents.