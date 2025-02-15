#-----------------Installierungen----------------------------
#Note: Be aware to have the latest version of python installed!

#get API Key for OpenAI: https://platform.openai.com/api-keys
#get API key for nomic: https://atlas.nomic.ai/data/<username>/org/keys
#pip install python-dotenv 
#pip install llama-index openai
#if the storage already exists, you cant load more docs into the rag pipeline
#pip install llama-index-vector-stores-faiss faiss-cpu
#pip install faiss-cpu
#pip install -U llama-index llama-index-embeddings-nomic
#pip install nest_asyncio

#-----------Hier startet der Code-----------------------------
#Import of libraries
from dotenv import load_dotenv
import os
import logging
import sys
import os.path
import faiss
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate,
    load_index_from_storage,
)

#Access api_key form .env-file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
nomic_api_key = os.getenv ("NOMIC_API_KEY")



embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=256,
    model_name="nomic-embed-text-v1.5",
)

llm = OpenAI(model="gpt-3.5-turbo")

#Change the llm-model & embedding-model(globally)
Settings.llm = llm
Settings.embed_model = embed_model

#Add logging to get a more detailed view on the running code
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__) #http request

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    logger.info("Erstelle neuen Index, da kein persistierter Speicher gefunden wurde...")
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()

    #Chun-zise and overlap can be adjusted
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    #FAISS based vectorstore
    d=256
    faiss_index = faiss.IndexFlatL2(d)  # 512 ist die Dimension des Embeddings, 1536 für gpt-3.5-embeddings
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex.from_documents(documents=documents, vector_store=vector_store)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    logger.info("Lade bestehenden Index aus dem persistierten Speicher...")
    
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
# Create a custom prompt template
template = """
    Dein Name ist "TutorGPT-2.0". Du bist ein motivierender Tutor für das Modul Distributed Systems. 
    Durch Erklärungen und Beantworten von Fragen hilfst du Studenten die Konzepte aus den Lerninhalten zu verstehen. 
    
    We have provided context information below:

    {context_str}

    Follow these rules:
    1. if the information is not sufficient, tell the user explicitly that the data is missing.
    2. do not make assumptions.
    3. answer concisely and clearly.
    4. be helpful
    5. be kind.
    6. provide credible answers with sources. Include chapters and page numbers if applicable.

    Answer the question: {query_str}
    """


qa_template = PromptTemplate(template)

query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_template)
response = query_engine.query("Wofür steht broadcast?")
print(response)
