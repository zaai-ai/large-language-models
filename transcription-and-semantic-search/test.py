from encoder.encoder import Encoder
import whisper
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
import utils
from whisperx import load_align_model, align

utils.download_youtube_video('https://www.youtube.com/watch?v=w5tWYmIOWGk')
utils.convert_to_wav('On Top Of The World.mp4')

model = whisper.load_model("base", "cpu")
result = model.transcribe("On Top Of The World.wav")

model_a, metadata = load_align_model(language_code=result['language'], device="cpu")
result_aligned = align(result['segments'], model_a, metadata, "On Top Of The World.wav", "cpu")

docs = [Document(page_content=f'start {item["start"]} - end {item["end"]}: {item["text"]}') for item in result_aligned['segments']]

encoder = Encoder()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
     host="localhost",
     port="5432",     
     database="postgres",
     user="admin",
     password="root",
)

COLLECTION_NAME = "test"

db = PGVector.from_documents(
    embedding=encoder.encoder,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

docs_with_score = db.similarity_search("quando é que falam do mundo?")