from typing import Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class DeduplicateRetriever(BaseRetriever):
    base_retriever:BaseRetriever

    def __init__(self, base_retriever:BaseRetriever):
        super(DeduplicateRetriever, self).__init__(base_retriever=base_retriever)
        
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        
        unique_documents = {}
        documents = []
        
        for doc in docs:
            content = doc.page_content.strip()
            if content in unique_documents:
                continue
            else:
                unique_documents[content] = True
                documents.append(doc)
  
        return documents