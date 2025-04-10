import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma


load_dotenv()

def get_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    print("\n [LOG] OK !! get_embeddings_model DONE \n")
    return embeddings
    

def get_legal_data_vector_store_retriever(embedding):
    # persistent_directory="D:/Data/Circle/BOT/VectorStores/legal_data_vector_store"
    persistent_directory="D:/Elana/fp_gen_ai/VectorStores/legal_data_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},)
    print("\n [LOG] OK !! get_legal_data_vector_store_retriever DONE \n")
    return retriever
    
    


def get_startup_data_vector_store_retriever(embedding):
    # persistent_directory="D:/Data/Circle/BOT/VectorStores/startup_data_vector_store"
    persistent_directory="D:/Elana/fp_gen_ai/VectorStores/startup_data_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},)
    
    print("\n [LOG] OK !! get_startup_data_vector_store_retriever DONE \n")
    return retriever


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    print("\n [LOG] OK !! get_llm DONE \n")
    return llm













# Load the existing Chroma vector store
# current_dir = os.path.dirname(os.path.abspath(__file__))
# db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
# persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# # Check if the Chroma vector store already exists
# if os.path.exists(persistent_directory):
#     print("Loading existing vector store...")
#     db = Chroma(persist_directory=persistent_directory,
#                 embedding_function=None)
# else:
#     raise FileNotFoundError(
#         f"The directory {persistent_directory} does not exist. Please check the path."
#     )

# Define the embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )

# Load the existing vector store with the embedding function
# db = Chroma(persist_directory=persistent_directory,
#             embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )

# Create a ChatOpenAI model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
