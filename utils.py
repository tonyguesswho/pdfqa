from langchain.vectorstores import Pinecone
from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def upload_to_pinecone(**kwargs):
    texts = kwargs.get("texts")
    embeddings = kwargs.get("embeddings")
    index_name = kwargs.get("index_name")
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name
    )
    return docsearch
