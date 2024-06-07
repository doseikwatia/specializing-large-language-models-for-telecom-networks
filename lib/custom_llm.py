from typing import Any, Dict, Iterator, List, Optional
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.language_models.llms import LLM



class CustomTransformersLLM(LLM):
    max_length :int    
    generator  :Any
    model_name: str
    
    def __init__(self, model_name: str, max_length:int):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token 
        model =  AutoModelForCausalLM.from_pretrained(model_name,)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map='auto')
        super(CustomTransformersLLM,self).__init__(max_length=max_length,generator = generator, model_name = model_name)


        
    @property
    def _llm_type(self) -> str:
        return "custom_transformers_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.generator(prompt, max_new_tokens=self.max_length, num_return_sequences=1, return_full_text=False,truncation=True)
        text = response[0]['generated_text']
        
        # Handle stop tokens if provided
        if stop is not None:
            for token in stop:
                text = text.split(token)[0]
                
        return text.strip()

    def stream(self, prompt: str, chunk_size: int = 50) -> Iterator[str]:
        """
        Stream the response in chunks.
        Args:
            prompt (str): The input prompt to generate text from.
            chunk_size (int): The size of each chunk to yield.
        Yields:
            Iterator[str]: Chunks of the generated text.
        """
        return [self._call(prompt)]
    
    


def main():
    model_name = 'microsoft/phi-2'
    max_length = 128
    llm = CustomTransformersLLM(model_name=model_name,max_length=max_length)
    response = llm('Provide a better search query for web search engine to answer the given question, end the queries with ’**’.  Question How does a supporting UE attach to the same core network operator from which it detached in a shared network? Answer:')
    print(response)

if __name__ == '__main__':
    main()