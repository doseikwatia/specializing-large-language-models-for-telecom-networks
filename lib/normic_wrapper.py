from typing import List
from langchain_core.embeddings.embeddings import Embeddings
from gpt4all import Embed4All


class NomicEmbedding(Embeddings):
    def __init__(self, model_name,dimensionality,**kwargs) -> None:
        super().__init__()
        self.__dimensionality = dimensionality
        self.__embeddings = Embed4All(model_name=model_name,**kwargs)

    def __embed_documents(self, texts: List[str], prefix) -> List[List[float]]:
        embeddings = [self.__embeddings.embed(text,prefix=prefix,dimensionality=self.__dimensionality) for text in texts]
        return [list(map(float, e)) for e in embeddings]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.__embed_documents(texts=texts,prefix='search_document')
    
    def embed_query(self, text: str) -> List[float]:
        return self.__embed_documents(texts=[text],prefix='search_query')[0]
