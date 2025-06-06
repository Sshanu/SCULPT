import llm
import ollama

def invoke_llm(model, system_prompt, user_prompt, max_tokens, temperature, top_p, index=-1):
    if model == "gpt4o":
        template = "<|im_start|>system\n???INSERT SYSTEM PROMPT HERE???\n<|im_end|>\n<|im_start|>user\n???INSERT USER PROMPT HERE???\n<|im_end|>\n<|im_start|>assistant"
        prompt = template.replace('???INSERT SYSTEM PROMPT HERE???', system_prompt)
        prompt = prompt.replace('???INSERT USER PROMPT HERE???', user_prompt)
        return llm.gpt4(prompt, model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    else:
        return ollama.ollama(user_prompt, system_prompt, model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, index=index)


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('## '):
            # first word without punctuation
            current_header = line[3:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result

def update_token_usage(token_usage, token_details):
    for key in token_details.keys():
        if key in token_usage.keys():
            token_usage[key] += token_details[key]
        else:
            token_usage[key] = token_details[key]
    return token_usage


