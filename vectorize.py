from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params

DATA_PATH = "data"

def main():
  documents = load_documents() # Step 1
  chunks = split_documents(documents) # Step 2

# Step 1: Load
  document_loader = PyPDFDirectoryLoader(DATA_PATH)
  documents = document_loader.load()

# Step 2: Transform (Split)
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=80,
      length_function=len,
      is_separator_regex=False,
  )
  docs = text_splitter.split_documents(documents)

# Step 3: Embed
  embedding = OpenAIEmbeddings(openai_api_key="")

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient("conn_string") # Mongo connection string
collection = client["db_name"]["collection_name"]

# Reset w/out deleting the Search Index
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
docsearch = MongoDBAtlasVectorSearch.from_documents(
  docs, embedding, collection=collection, index_name="index_name"
)