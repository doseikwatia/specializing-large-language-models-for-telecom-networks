from langchain_core.prompts import PromptTemplate

train_response_template = "### Answer:"
train_instruction_template = "### Instructions:"

def get_training_prompt(question):
    prompt_template = PromptTemplate.from_template("""{{ instructions }} 
Based on only the provided context, select the correct answer from the choices given. Provide your answer in the following format: option Number) Answer. Do not include any additional text or explanation.

Context:
{{ context }}

Question:
{{ question }}

Choices:
{% for option in options %}{{ option[0] }}) {{option[1]}}
{% endfor %}

{{ answer }}

""",template_format="jinja2")
    prompt_context = {
        'context': question['explanation'],
        'question': question['question'],
        'instructions': train_instruction_template,
        'answer': train_response_template,
        'options':  list(filter(lambda item: item[0].startswith("option")  ,question.items()))
    }
    answer = question['answer']
    prompt = prompt_template.invoke(prompt_context).to_string()
    return {'question': prompt,'answer': answer}

def get_inference_prompt(question, context):
    new_question = dict(question)
    new_question['explanation'] = context 
    return get_training_prompt(new_question)