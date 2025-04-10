import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma


load_dotenv()
def get_embeddings_model():
    # Explicitly set the device
    device = "cpu"  # Use CPU as it's safer for compatibility
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    print("\n [LOG] OK !! get_embeddings_model DONE \n")
    return embeddings
# def get_embeddings_model():
#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2")
#     print("\n [LOG] OK !! get_embeddings_model DONE \n")
#     return embeddings
    

def get_legal_data_vector_store_retriever(embedding):
    # persistent_directory="D:/Data/Circle/BOT/VectorStores/legal_data_vector_store"
    persistent_directory="./VectorStores/legal_data_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},)
    print("\n [LOG] OK !! get_legal_data_vector_store_retriever DONE \n")
    return retriever
    
    


def get_startup_data_vector_store_retriever(embedding):
    # persistent_directory="D:/Data/Circle/BOT/VectorStores/startup_data_vector_store"
    persistent_directory="./VectorStores/startup_data_vector_store"
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



