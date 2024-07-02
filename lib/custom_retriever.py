from typing import Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from lib.prompt import get_qa_inference_prompt

class CustomRetriever(BaseRetriever):
    base_retriever:BaseRetriever
    oracle:Any
    max_new_tokens:int
    def __init__(self, base_retriever:BaseRetriever,oracle:Any,max_new_tokens=32):
        super(CustomRetriever, self).__init__(base_retriever=base_retriever,oracle = oracle,max_new_tokens=max_new_tokens)
        
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        context =  (' '.join(list(map(lambda d:d.page_content,docs)))).replace('\n', '. ')
        prompt = get_qa_inference_prompt({'question':query,},context)['prompt']
        page_content = self.oracle(prompt,max_new_tokens=self.max_new_tokens,return_full_text=False)
        return [Document(page_content[0]['generated_text'],)]