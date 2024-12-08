#-----------------Installierungen----------------------------
#Note: Be aware to have the latest version of python installed!

#get API Key for OpenAI: https://platform.openai.com/api-keys
#get API key for nomic: https://atlas.nomic.ai/data/<username>/org/keys
#pip install python-dotenv 
#pip install llama-index openai
#if the storage already exists, you cant load more docs into the rag pipeline
#pip install llama-index-vector-stores-faiss faiss-cpu
#pip install faiss-cpu
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
from llama_index.embeddings.openai import OpenAIEmbedding
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


embed_model = OpenAIEmbedding(model="text-embedding-3-small")


llm = OpenAI(model="gpt-4-turbo")

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
    Settings.chunk_size = 250
    Settings.chunk_overlap = 25

    #FAISS based vectorstore
    d=1536
    faiss_index = faiss.IndexFlatL2(d)  # 512 ist die Dimension des Embeddings, 1536 für gpt-3.5-embeddings
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    print(f"Die Dimension des FAISS-Vektorraums ist: {faiss_index.d}")

    index = VectorStoreIndex.from_documents(documents=documents, vector_store=vector_store)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    logger.info("Lade bestehenden Index aus dem persistierten Speicher...")
    
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


#Create a custom prompt template
template = """  
    Your name is "TutorGPT2.0“. You are an upbeat and encouraging tutor for Distributed Systems. 
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


qa_template = PromptTemplate(template)

#Query our index
query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_template)
response = query_engine.query("Nenne mir die Dimensionsanzahl von dem FAiss Vectorstore, welchen du verwendest")
print(response)

#embeddings = embed_model.get_text_embedding(
#    "Open AI new Embeddings models is awesome."
#)
#
#print(len(embeddings))
