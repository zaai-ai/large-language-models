import yaml

from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path

class Encoder:
    """ Encoder to create workds embeddings from text """
    def __init__(self) -> None:
        self.config = yaml.safe_load(open(f'{Path().parent.absolute()}/config.yaml'))
        self.encoder = HuggingFaceEmbeddings(
            model_name=self.config['encoder']['model_path'],
            model_kwargs=self.config['encoder']['model_kwargs'],
            encode_kwargs=self.config['encoder']['encode_kwargs']
        )
