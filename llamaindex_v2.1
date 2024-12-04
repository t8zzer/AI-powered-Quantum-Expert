#-----------------Installierungen----------------------------
#Note: Be aware to have the latest version of python installed!

#pip install 
#python-dotenv 
#llama-index openai
#if the storage already exists, you cant load more docs into the rag pipeline

#-----------Hier startet der Code-----------------------------
#Import of libraries
from dotenv import load_dotenv
import os
import logging
import sys
import os.path
import faiss
from llama_index.llms.openai import OpenAI
#from llama_index.embeddings.openai import OpenAIEmbedding
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

#Change the llm-model (globally)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

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
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    #FAISS based vectorstore
    d=1536
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
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question as a tutor: {query_str}\n"
)

qa_template = PromptTemplate(template)

query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_template)
response = query_engine.query("I would like you to summarize the following pdf: elections.pdf?")
print(response)