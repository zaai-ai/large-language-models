from langchain.docstore.document import Document
from langchain.document_loaders import HuggingFaceDatasetLoader

from encoder.encoder import Encoder
from generator.generator import Generator
from retriever.vector_db import VectorDatabase

# load dataset
loader = HuggingFaceDatasetLoader("luisroque/instruct-python-llama2-20k", "text")
docs = loader.load()
train = [
    Document(page_content=x.page_content.split("[/INST]")[1]) for x in docs[:-1000]
]
test = [Document(page_content=x.page_content.split("[/INST]")[0]) for x in docs[-1000:]]

TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
{context}
Question: {question}
Answer:
"""

encoder = Encoder()
faiss_db = VectorDatabase()
generator = Generator(TEMPLATE)

passages = faiss_db.create_passages_from_documents(train)
faiss_db.store_passages_db(passages, encoder.encoder)

query = test[0].page_content.split("<</SYS>>")[2]
context = faiss_db.retrieve_most_similar_document(query)
print(generator.get_answer(context, query))

print(query)
print(context)
print(docs[-1000].page_content.split("[/INST]")[1])
