import os
import pickle
from PyPDF2 import PdfReader
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from transformers import GPT2Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

os.environ.get("OPENAI_API_KEY")

st.title("🦜🔗 AI-powered PDF Analyzer")
st.markdown("""
This application uses state-of-the-art language models to summarize PDF files and answer questions about the content. 
""")
st.image('ai.gif')

with st.sidebar:
    st.header("Controls")
    st.markdown("""
    This application can summarize at least 4 pages of a PDF. However, by providing your OpenAI API key, you can summarize more.
    """)
    api_key = st.text_input("Enter your OpenAI API key:")

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    else:
        os.environ['OPENAI_API_KEY'] = my_api_key

    uploaded_file = st.file_uploader(
        "Please upload your PDF file here", type=["pdf"])
    query = st.text_input("Ask questions about the PDF:")

if 'summary_output' not in st.session_state:
    st.session_state['summary_output'] = None

if uploaded_file is not None:
    # Save the uploaded file to the upload directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully.")

    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

    with st.expander("Extracted Text"):
        st.write(extracted_text)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=5000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.create_documents([extracted_text])

    try:
        if st.session_state['summary_output'] is None:
            if len(texts) < 4:
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokens = tokenizer.encode(
                    extracted_text, add_special_tokens=False)
                num_tokens = len(tokens)

                llm = ChatOpenAI(temperature=0)
                map_prompt = """
                Feel Free to use analogies
                Write a concise summary in simple terms of the following:
                "{text}"
                CONCISE SUMMARY:
                """
                map_prompt_template = PromptTemplate(
                    template=map_prompt, input_variables=["text"])

                sumarry_chain = load_summarize_chain(
                    llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template)

                with st.spinner('Generating Summary...'):
                    st.session_state['summary_output'] = sumarry_chain.run(
                        texts)

        st.markdown(f"## Summary of {uploaded_file.name}:")
        st.write(st.session_state['summary_output'])

        # embeddings
        chunks = text_splitter.split_text(text=extracted_text)
        store_name = uploaded_file.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                pickle.dump(VectorStore, f)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = ChatOpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with st.spinner('Answering your question...'):
                response = chain.run(input_documents=docs, question=query)
            st.markdown(f"## Answer for \"{query}\":")
            st.write(response)

    except Exception as e:
        st.error(
            "There seems to be a problem with your OpenAI API key. Please check and try again.")
