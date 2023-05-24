import os

import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import upload_to_pinecone


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Document")
    st.header("Q/A with your personl Document")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

    pdf_url = st.text_input("Provide PDF URL:")
    # e.g
    # http://www.denosa.org.za/DAdmin/upload/news/History%20of%20iPhone%20-%20Wikipedia.pdf
    # "https://arxiv.org/pdf/2302.03803.pdf"

    if pdf_url:
        # Load your data
        loader = PyPDFLoader(pdf_url)

        data = loader.load()

        st.write(f"You have {len(data)} document(s) in your data")
        st.write(f"There are {len(data[2].page_content)} characters in your document")

        # Chunk your data up into smaller documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        # print(f"Now you have {len(texts)} documents")

        # Create embeddings of your documents to get ready for semantic search
        print("create embeddings")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        print("saving in vector DB")
        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_API_ENV,  # next to api key in console
        )
        index_name = "pdfqa"

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docsearch = upload_to_pinecone(
                texts=texts, embeddings=embeddings, index_name=index_name
            )

            # "What are Orbifolds"
            docs = docsearch.similarity_search(user_question)

            # get similar docs

            # Answer in Natural language
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == "__main__":
    main()
