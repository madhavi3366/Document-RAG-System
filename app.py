import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Document Search Engine",
    page_icon="📂",
    layout="wide"
)

# ---------------- LOGIN AUTHENTICATION ----------------

USER_CREDENTIALS = {
    "admin": "admin123",
    "madhavi": "password123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Secure Login")

    st.markdown("Please login to access the Document Search System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if (
            username in USER_CREDENTIALS and
            USER_CREDENTIALS[username] == password
        ):

            st.session_state.logged_in = True
            st.success("Login Successful!")

            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()

# ---------------- HELPER FUNCTION ----------------

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📂 Search Engine")

    st.markdown("""
    ### Features

    ✅ Secure Login

    ✅ File Name Search

    ✅ Semantic Search

    ✅ Document Question Answering

    ✅ Download Documents

    ✅ Search History
    """)

    st.markdown("---")

    st.success("Logged in as User")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.rerun()

    if st.button("🗑 Clear History"):
        st.session_state.history = []

# ---------------- TITLE ----------------

st.title("📂 AI-Powered Document Search Engine")

st.markdown("""
Search documents using:

- **File Name Search**
- **Semantic Search**
- **Ask Questions from Documents**
""")

# ---------------- LOAD VECTOR DB ----------------

@st.cache_resource
def load_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db = load_vector_db()

# ---------------- SEARCH BAR ----------------

query = st.text_input(
    "🔍 Search by file name or document content"
)

# ---------------- SEARCH ----------------

if query:

    st.session_state.history.append(query)

    # -------- FILE NAME SEARCH --------

    st.subheader("📁 File Name Matches")

    filename_matches = []

    for root, dirs, files in os.walk("documents"):

        for file in files:

            if normalize(query) in normalize(file):

                filename_matches.append(
                    os.path.join(root, file)
                )

    if filename_matches:

        for path in filename_matches:

            file = os.path.basename(path)

            with st.expander(f"📄 {file}"):

                st.success("Matching file found")

                st.write(f"**File Name:** {file}")

                with open(path, "rb") as f:

                    st.download_button(
                        label="⬇ Download File",
                        data=f,
                        file_name=file,
                        key=file
                    )

    else:
        st.info("No filename matches found.")

    # -------- SEMANTIC SEARCH --------

    st.subheader("🧠 Semantic Search Results")

    results = db.similarity_search_with_score(
        query,
        k=5
    )

    displayed_files = set()

    if results:

        for doc, score in results:

            filename = doc.metadata.get(
                "filename",
                "Unknown"
            )

            if filename not in displayed_files:

                displayed_files.add(filename)

                with st.expander(f"📄 {filename}"):

                    st.write("### Preview")

                    st.write(doc.page_content[:700])

                    st.write(
                        f"**Similarity Score:** {round(score, 2)}"
                    )

                    file_path = doc.metadata.get(
                        "filepath",
                        ""
                    )

                    if os.path.exists(file_path):

                        with open(file_path, "rb") as f:

                            st.download_button(
                                label=f"⬇ Download {filename}",
                                data=f,
                                file_name=filename,
                                key=f"download_{filename}"
                            )

    else:
        st.warning("No semantic matches found.")

# ---------------- QUESTION ANSWERING ----------------

st.markdown("---")

st.subheader("💬 Ask Questions About Your Documents")

question = st.text_input(
    "Example: How many sales happened?"
)

if question:

    with st.spinner("Generating answer..."):

        retriever = db.as_retriever(
            search_kwargs={"k": 5}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain.invoke(
            {"query": question}
        )

        st.subheader("🤖 Answer")

        st.success(response["result"])

        st.subheader("📚 Source Documents")

        displayed_sources = set()

        for doc in response["source_documents"]:

            filename = doc.metadata.get(
                "filename",
                "Unknown"
            )

            if filename not in displayed_sources:

                displayed_sources.add(filename)

                with st.expander(f"📄 {filename}"):

                    st.write(doc.page_content[:700])

# ---------------- SEARCH HISTORY ----------------

if st.session_state.history:

    st.markdown("---")

    st.subheader("📝 Recent Searches")

    for item in reversed(
            st.session_state.history[-5:]):

        st.write("•", item)