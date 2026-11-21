import time
import os
import glob

import streamlit as st
import numpy as np
import pypdf
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from google import genai


st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="💼",
    layout="centered"
)


@st.cache_resource
def load_rag_pipeline():

    pdf_folder = "zyro-dynamics-hr-corpus"

    pdf_files = sorted(
        glob.glob(
            os.path.join(pdf_folder, "*.pdf")
        )
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No HR policy PDF files were found in "
            "'zyro-dynamics-hr-corpus'."
        )

    documents = []

    for pdf_file in pdf_files:

        reader = pypdf.PdfReader(pdf_file)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text and text.strip():

                documents.append({
                    "text": text.strip(),
                    "source": os.path.basename(pdf_file),
                    "page": page_number
                })


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for doc in documents:

        split_texts = text_splitter.split_text(
            doc["text"]
        )

        for text in split_texts:

            chunks.append({
                "text": text,
                "source": doc["source"],
                "page": doc["page"]
            })


    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    embeddings_array = np.array(
        embeddings
    )

    dimension = embeddings_array.shape[1]

    faiss_index = faiss.IndexFlatIP(
        dimension
    )

    faiss_index.add(
        embeddings_array.astype("float32")
    )


    def retrieve_chunks(
        question,
        top_k=5
    ):

        question_embedding = embedding_model.encode(
            [question]
        ).astype("float32")

        scores, indices = faiss_index.search(
            question_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            results.append({
                "text": chunks[index]["text"],
                "source": chunks[index]["source"],
                "page": chunks[index]["page"],
                "score": float(score)
            })

        return results


    # Get Google API key
    api_key = None

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get(
            "GOOGLE_API_KEY"
        )

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )


    client = genai.Client(
        api_key=api_key
    )


    def retrieve_and_build_context(
        question
    ):

        results = retrieve_chunks(
            question,
            top_k=5
        )

        context = "\n\n".join(
            f"Source: {r['source']}, "
            f"Page: {r['page']}\n"
            f"{r['text']}"
            for r in results
        )


        # Keep only unique source + page combinations
        unique_sources = []

        seen_sources = set()

        for r in results:

            source_key = (
                r["source"],
                r["page"]
            )

            if source_key not in seen_sources:

                seen_sources.add(
                    source_key
                )

                unique_sources.append({
                    "source": r["source"],
                    "page": r["page"]
                })


        return {
            "question": question,
            "context": context,
            "sources": unique_sources
        }
    def generate_grounded_answer(
    data
):

    prompt = f"""
You are the HR Help Desk assistant for Zyro Dynamics.

Answer the employee's question using ONLY the HR policy
information provided in the context.

Rules:
- Give a clear and concise answer.
- Do not invent information.
- Do not use outside knowledge.
- Base the answer only on the provided HR policy context.
- If the context does not contain enough information to answer,
  politely say that the information is not available in the HR
  policy documents.
- For questions unrelated to Zyro Dynamics HR policies, politely
  state that you can only help with Zyro Dynamics HR policy queries.

Context:
{data["context"]}

Employee question:
{data["question"]}

Answer:
"""

    response = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            break

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(5)
            else:
                raise

    return {
        "answer": response.text.strip(),
        "sources": data["sources"]
    }
    # LCEL RAG chain
    rag_chain = (
        RunnableLambda(
            retrieve_and_build_context
        )
        | RunnableLambda(
            generate_grounded_answer
        )
    )


    return rag_chain


# Initialize RAG system
try:

    rag_chain = load_rag_pipeline()

except Exception as e:

    st.error(
        "Unable to initialize the HR RAG system."
    )

    st.error(str(e))

    st.stop()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title(
    "💼 Zyro Dynamics HR Assistant"
)

st.write(
    "Ask questions about Zyro Dynamics HR policies."
)


question = st.text_input(
    "Enter your HR policy question:",
    placeholder=(
        "e.g. How much Earned Leave can I "
        "carry forward?"
    )
)


if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching HR policies..."
        ):

            try:

                result = rag_chain.invoke(
                    question
                )


                # Display answer
                st.subheader(
                    "Answer"
                )

                st.write(
                    result["answer"]
                )


                # Display sources
                if result.get("sources"):

                    st.subheader(
                        "Sources"
                    )

                    for source in result["sources"]:

                        st.write(
                            f"📄 "
                            f"{source['source']} "
                            f"— Page "
                            f"{source['page']}"
                        )


            except Exception as e:

                st.error(
                    "Sorry, I was unable to process "
                    "your question right now."
                )

                st.error(
                    str(e)
                )
