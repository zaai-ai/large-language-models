from encoder.encoder import Encoder
from generator.generator import Generator
from langchain.document_loaders import HuggingFaceDatasetLoader
from retriever.vector_db import VectorDatabase

# load dataset
loader = HuggingFaceDatasetLoader("luisroque/instruct-python-500k", 'answer')
docs = loader.load()[:1000]

TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
{context}
Question: {question}
Answer:
"""

encoder = Encoder()
faiss_db = VectorDatabase()
generator = Generator(TEMPLATE)

passages = faiss_db.create_passages_from_documents(docs)
faiss_db.store_passages_db(passages, encoder.encoder)

from datasets import load_dataset
data = load_dataset("luisroque/instruct-python-500k")
query = data['train'][3]['question']

context = faiss_db.retrieve_most_similar_document(query)
generator.define_prompt()
print(generator.get_answer(context, query))

print(query)
print(data['train'][3]['answer'])
print(context)