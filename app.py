import os
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

load_dotenv()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

os.environ.get("OPENAI_API_KEY")

st.title("🦜🔗 Summary of your PDF")
st.image('ai.gif')

uploaded_file = st.file_uploader(
    "Please upload your PDF file here", type=["pdf"])


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
        texto = extracted_text

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=10000,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.create_documents([extracted_text])

    # st.write(texts[0])
    # st.write(len(texts))

    if len(texts) < 4:

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(extracted_text, add_special_tokens=False)
        num_tokens = len(tokens)

        st.write("Number of tokens:", num_tokens)

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

        with st.spinner('Loading...'):
            output = sumarry_chain.run(texts)
        st.write(output)

    else:
        st.write("You got a bit text")
