import os
import re
import pandas as pd
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

    st.markdown(
        "Please login to access the Document Search System"
    )

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

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

# ---------------- HELPER FUNCTIONS ----------------

def normalize(text):
    return re.sub(
        r'[^a-zA-Z0-9]',
        '',
        text.lower()
    )


def answer_csv_question(question):

    try:

        csv_files = []

        # Get all CSV files
        for file in os.listdir("documents"):

            if file.endswith(".csv"):

                csv_files.append(
                    os.path.join("documents", file)
                )

        if len(csv_files) == 0:
            return None

        # Read first CSV
        df = None

        for csv_file in csv_files:
            temp_df = pd.read_csv(csv_file)
            cols = [c.lower() for c in temp_df.columns]
            if "gender" in cols or "sex" in cols:
                df = temp_df
                break
            if df is None:
                df = pd.read_csv(csv_files[0])

        # Show columns in sidebar
        st.sidebar.markdown("### CSV Columns")
        st.sidebar.write(df.columns.tolist())

        q = question.lower()

        # ---------------- TOTAL RECORDS ----------------

        if (
            "how many" in q and
            (
                "records" in q or
                "rows" in q or
                "entries" in q
            )
        ):

            return f"📊 Total Records: {len(df)}"

        # ---------------- TOTAL SALES ----------------

        if "how many sales" in q:

            return f"💰 Total Sales Records: {len(df)}"

        # ---------------- MALE/FEMALE COUNT ----------------

        if "male" in q and "female" in q:

            gender_col = None

            for col in df.columns:

                col_name = col.lower().strip()

                if (
                    "gender" in col_name or
                    "sex" in col_name
                ):

                    gender_col = col
                    break

            if gender_col:

                counts = (
                    df[gender_col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .value_counts()
                )

                male_count = 0
                female_count = 0

                for key, value in counts.items():

                    if key in ["male", "m"]:
                        male_count += value

                    elif key in ["female", "f"]:
                        female_count += value

                return (
                    f"👨 Male Count: {male_count}\n\n"
                    f"👩 Female Count: {female_count}"
                )

            else:

                return (
                    "❌ Gender column not found.\n\n"
                    f"Available Columns:\n"
                    f"{', '.join(df.columns)}"
                )

        # ---------------- AVERAGE ----------------

        if "average" in q:

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            if len(numeric_cols) > 0:

                col = numeric_cols[0]

                avg = round(
                    df[col].mean(),
                    2
                )

                return (
                    f"📈 Average value of "
                    f"'{col}' = {avg}"
                )

        # ---------------- TOTAL OF A COLUMN ----------------

        if "total" in q:

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            for col in numeric_cols:

                if col.lower() in q:

                    total = round(df[col].sum(), 2)

                    return (
                        f"📊 Total {col}: {total}"
                    )

        # ---------------- MAX VALUE ----------------

        if "maximum" in q or "highest" in q:

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            if len(numeric_cols) > 0:

                col = numeric_cols[0]

                return (
                    f"📈 Highest {col}: "
                    f"{df[col].max()}"
                )

        # ---------------- MIN VALUE ----------------

        if "minimum" in q or "lowest" in q:

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            if len(numeric_cols) > 0:

                col = numeric_cols[0]

                return (
                    f"📉 Lowest {col}: "
                    f"{df[col].min()}"
                )

        return None

    except Exception as e:

        return f"CSV Error: {str(e)}"


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

    ✅ CSV Analytics

    ✅ Document Question Answering

    ✅ Download Documents

    ✅ Search History
    """)

    st.markdown("---")

    st.success("Logged in Successfully")

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

                with open(path, "rb") as f:

                    st.download_button(
                        label="⬇ Download File",
                        data=f,
                        file_name=file,
                        key=file
                    )

    else:
        st.info("No filename matches found.")

    # Semantic Search

    st.subheader("🧠 Semantic Search Results")

    results = db.similarity_search_with_score(
        query,
        k=5
    )

    displayed_files = set()

    for doc, score in results:

        filename = doc.metadata.get(
            "filename",
            "Unknown"
        )

        if filename not in displayed_files:

            displayed_files.add(filename)

            with st.expander(f"📄 {filename}"):

                st.write(
                    doc.page_content[:700]
                )

                st.write(
                    f"Similarity Score: "
                    f"{round(score,2)}"
                )

# ---------------- QUESTION ANSWERING ----------------

st.markdown("---")

st.subheader(
    "💬 Ask Questions About Your Documents"
)

question = st.text_input(
    "Example: How many males and females are there?"
)

if question:

    # Try CSV analysis first

    csv_answer = answer_csv_question(
        question
    )

    if csv_answer:

        st.subheader("🤖 Answer")

        st.success(csv_answer)

    else:

        with st.spinner(
                "Generating answer..."):

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

            st.success(
                response["result"]
            )

            st.subheader(
                "📚 Source Documents"
            )

            shown = set()

            for doc in response[
                    "source_documents"]:

                filename = doc.metadata.get(
                    "filename",
                    "Unknown"
                )

                if filename not in shown:

                    shown.add(filename)

                    with st.expander(
                            f"📄 {filename}"):

                        st.write(
                            doc.page_content[:700]
                        )

# ---------------- HISTORY ----------------

if st.session_state.history:

    st.markdown("---")

    st.subheader("📝 Recent Searches")

    for item in reversed(
            st.session_state.history[-5:]):

        st.write("•", item)