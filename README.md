
# pdfqa
#This is a Python application that allows you to load a Documet and ask questions about it using natural language.
# Technologies used
- python
- Langchain
- OpenAI
- Pinecone
- Streamlit

# How it works
- Reads pdf file from provided url
- Splits the text into smaller chunks that can be then fed into a LLM
- Create vector representations usingOpenAI embeddings
- Finds the chunks that are semantically similar to provided question
- Feeds those chunks to the LLM to generate a natural response


# Installation
- clone this repository and install the requirements:

- pip install -r requirements.txt
- Provided ENV variables following the example in env.example file


# Running application
Run using `streamlit run app.py`

<img width="843" alt="Screenshot 2023-05-24 at 18 01 52" src="https://github.com/tonyguesswho/pdfqa/assets/19865565/3b53c68b-e969-4498-b246-d2a1068ff688">
