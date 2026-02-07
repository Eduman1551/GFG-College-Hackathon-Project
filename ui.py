import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Academic Second Brain", layout="centered")

st.title("📚 Intelligent Academic Second Brain")
st.write("Upload notes or ask questions from your academic content.")

# -----------------------------
# PDF Upload Section
# -----------------------------
st.header("📄 Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload and Index"):
        files = {"file": uploaded_file}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)

        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error("Upload failed")

# -----------------------------
# Question Asking Section
# -----------------------------
st.header("❓ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        payload = {"question": question}
        response = requests.post(f"{BACKEND_URL}/ask", json=payload)

        if response.status_code == 200:
            data = response.json()
            st.subheader("✅ Answer")
            st.write(data["answer"])

            st.subheader("📌 Sources")
            for src in data["sources"]:
                st.write(f"- {src}")
        else:
            st.error("Error getting response")  