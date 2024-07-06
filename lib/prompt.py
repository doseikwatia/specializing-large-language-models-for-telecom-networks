from langchain_core.prompts import PromptTemplate

train_response_template = "### Answer:"
train_instruction_template = "### Instructions:"
train_explanation_template = "### Explanation:"

def get_mcq_prompt(question, context, show_answers):
    prompt_template = PromptTemplate.from_template("""{{ instructions_tag }} 
Based on the provided context, select the correct answer from the choices given. Provide your answer in the following format: option Number: Answer.

Context:
{{ context }}

Question:
{{ question }}
Choices:
{% for option in options %}{{ option[0] }}: {{option[1]}}
{% endfor %}
{{ answer_tag }}{% if show_answers %}
{{ answer }}
{% endif %}
""",template_format="jinja2")
    prompt_context = {
        'context':  context,
        'question': question['question'],
        'instructions_tag': train_instruction_template,
        'answer_tag': train_response_template,
        'show_answers': show_answers,
        'answer': question.get("answer", ''),
        'options':  list(filter(lambda item: item[0].startswith("option")  ,question.items()))
    }
    answer = question.get("answer", None)
    explanation = question.get("explanation", None)
    prompt = prompt_template.invoke(prompt_context).to_string()
    return {'prompt': prompt,'answer': answer, 'explanation': explanation}


def get_qa_prompt(question, context, show_answers):
    prompt_template = PromptTemplate.from_template("""{{ instructions_tag }} 
Based on the provided context, answer the question.

Context:
{{ context }}

Question:
{{ question }}

{{ answer_tag }}{% if show_answers %}
{{ answer }}
{% endif %}
""",template_format="jinja2")
    prompt_context = {
        'context': context,
        'question': question['question'],
        'instructions_tag': train_instruction_template,
        'answer_tag': train_response_template,
        'show_answers': show_answers,
        'answer': question.get("explanation", ''),
    }
    answer = question.get("answer", None)
    explanation = question.get("explanation", None)
    prompt = prompt_template.invoke(prompt_context).to_string()
    return {'prompt': prompt,'answer': answer, 'explanation': explanation}




def get_mcq_training_prompt(question, context):
    return get_mcq_prompt(question, context, True)

def get_mcq_inference_prompt(question, context):
    return get_mcq_prompt(question,context, False)

def get_qa_training_prompt(question, context):
    return get_qa_prompt(question, context, True)

def get_qa_inference_prompt(question, context):
    return get_qa_prompt(question,context, False)