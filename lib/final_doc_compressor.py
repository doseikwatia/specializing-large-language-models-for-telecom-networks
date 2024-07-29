from typing import Any, List, Optional, Sequence, Union
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_community.llms.llamafile import Llamafile
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class FinalDocumentCompressor(BaseDocumentCompressor):
    llm:Llamafile
    prompt: PromptTemplate
    max_word_count:int
    
    def __init__(self,n_predict = 512,seed = 1):
        llm = Llamafile(n_predict=n_predict, seed=seed,temperature=0.0)
        prompt =PromptTemplate.from_template('''<|system|>
You are a telecommunication engineering expert. Given the context, explain the correct answer for the following question succinctly. Please provide a response that is clear and concise, not exceeding {max_word_count} words.</s>
<|user|>
Context:
{context}
Question:
{question}</s>
<|assistant|>
'''
        )
        super(FinalDocumentCompressor, self).__init__(llm=llm, prompt=prompt,max_word_count=n_predict)
    
    def compress_documents(self,documents: Sequence[Document], query: str, callbacks = None) -> Sequence[Document]:
        context = ' '.join(list(map(lambda d: d.page_content,documents)))
        llm_input = self.prompt.invoke({
            'context':context,
            'question':query,
            'max_word_count': self.max_word_count
        })
        page_content = self.llm.invoke(llm_input)
        page_content = page_content.replace('</s>','').replace('<|system|>','').replace('<|ed|>','').replace('<|eot_id|>','').replace('<|assistant|>','')
        return [Document(page_content=page_content)]