import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import openai
import os
import altair as alt


openai.api_key = "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"

def main():
    st.header("Chat with One or more file's 💬")

    # Upload a PDF file
    files = st.file_uploader("Please upload your PDF", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)
    
    if files is not None:
        for file in files:
            file_extension = file.name.split('.')[-1].lower()

            if file_extension in ['csv', 'txt', 'pdf']:
                text = ""
                if file_extension == 'pdf':
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    text = file.read().decode('utf-8')
    
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                store_name = file.name[:-4]

                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", 'rb') as f:
                        VectorStore = pickle.load(f)
                else:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", 'wb') as f:
                        pickle.dump(VectorStore, f)

                query = st.text_input(f"What's your query about the {file.name}?")

                if query:
                    docs = VectorStore.similarity_search(query=query, k=3)
                    llm = OpenAI(model_name='gpt-3.5-turbo-0613', openai_api_key=openai.api_key)  # Pass the API key here
                    chain = load_qa_chain(llm=llm, chain_type='stuff')
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                    st.write(f"Results for {file.name}:")
                    st.write(response)

if __name__ == '__main__':
    main()
