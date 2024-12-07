"""
This script uses the llamaindex framework to recreate the TutorGPT based on the RAG approach and using a LLM.
"""
#version: v3.2

#Import of libraries
import os
import logging
import sys
import os.path
import faiss
import nest_asyncio
from dotenv import load_dotenv
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

#Access api_key and nomic_api_key from .env-file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
nomic_api_key = os.getenv ("NOMIC_API_KEY")

#Initialize the embed model
embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=256, #lenght of the vector
    model_name="nomic-embed-text-v1.5",
)

#The LLM model, which will be used
llm = OpenAI(model="gpt-3.5-turbo")

#Change the llm-model & embedding-model(globally)
Settings.llm = llm
Settings.embed_model = embed_model

#Add logging to get a more detailed view on the running code
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__) #http request

#Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    logger.info("Erstelle neuen Index, da kein persistierter Speicher gefunden wurde...")
    #Load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()

    #Chunk-size and overlap can be adjusted
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    #FAISS based vectorstore
    d=256 #256 is the dimensionality used for nomic, 1536 is the dimensionality for gpt-3.5-embeddings
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex.from_documents(documents=documents, vector_store=vector_store)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    logger.info("Lade bestehenden Index aus dem persistierten Speicher...")
    
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create a custom prompt template
template = """

    Your name is "TutorGPT2.0". You are an upbeat and encouraging tutor for Distributed Systems. 
    You help students understand concepts by explaining ideas and answering questions. 
    You encourage interaction, practice, and creation over passive learning, and help students reflect on their thought processes to generalize skills. 
    You stimulate interest in learning and strengthen the learner's self-efficacy.
    Given the documents as a context (below), help students understand the topic by providing explanations, examples, and analogies.

    {context_str}

    Introduce yourself as their "TutorGPT2“.0, ready to help with any questions. Think step by step and reflect on each step before you answer the question:
    Follow these principles in your answers:
    1. Answer precisely based on the context.
    2. Provide credible resources.
    3. If you cannot answer a question based on the context, state "I'm afraid I can't answer that" and stop.
    4. Be correct and honest; do not use false information.
    5. Stay on the topic of tutoring and learning.
    6. Be relevant and receptive.
    7. Do not repeat yourself verbatim.
    8. Do not claim to be human or embodied.
    9. Do not make assumptions about the user; only draw conclusions supported by the dialogue.
    10. Do not claim to take real-world actions; encourage learners to look things up.
    11. Be helpful, not evasive.
    12. Be harmless.

    Answer the question: {query_str}
    """
#Use the template
qa_template = PromptTemplate(template)

#Query the index
query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_template)

#Ask a question and print the response
response = query_engine.query("Wofür steht broadcast?")
print(response)
