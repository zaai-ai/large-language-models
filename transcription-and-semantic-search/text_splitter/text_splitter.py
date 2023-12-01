from base.config import Config
from langchain.text_splitter import CharacterTextSplitter


class TextSplitter(Config):
    """Text Splitter"""

    def __init__(self) -> None:
        super().__init__()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config["retriever"]["passage"]["chunk_size"],
            chunk_overlap=self.config["retriever"]["passage"]["chunk_overlap"],
        )

    def create_passages_from_documents(self, documents: list) -> list:
        """
        Splits the documents into passages of a certain length
        Args:
            documents (list): list of documents
        Returns:
            list: list of passages
        """
        return self.text_splitter.split_documents(documents)
