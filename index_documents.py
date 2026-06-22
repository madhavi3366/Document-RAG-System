import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCUMENT_FOLDER = "documents"

all_documents = []

for file in os.listdir(DOCUMENT_FOLDER):

    path = os.path.join(DOCUMENT_FOLDER, file)

    print("Processing:", file)

    try:

        # PDF Loader
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)

        # CSV Loader
        elif file.endswith(".csv"):
            loader = CSVLoader(path)

        else:
            print(f"Skipping {file}")
            continue

        docs = loader.load()

        # Store filename in metadata
        for doc in docs:
            doc.metadata["filename"] = file

        all_documents.extend(docs)

        print(f"{file} loaded successfully")

    except Exception as e:
        print(f"Error loading {file}: {e}")

print("Total Documents Loaded:", len(all_documents))

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print("Total Chunks:", len(chunks))

if len(chunks) == 0:
    raise ValueError("No document chunks found.")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
db = FAISS.from_documents(chunks, embeddings)

db.save_local("vector_store")

print("Vector store created successfully!")