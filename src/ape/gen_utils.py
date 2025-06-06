import re
import string
import random
import llm
import llm_utils

def read_prompt(file):
    with open(file, "r") as file:
        prompt = file.read()
        return prompt

# GPT4 Response Generation
def generate_response(input, gpt_model, args):
    response, usage = llm.gpt4(input, model=gpt_model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)[1]
    response = remove_extra_newlines(response)  
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    return response, usage

# Define a function to generate a random ID consisting of letters and digits
def generate_random_id():
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(32)])
    
def remove_extra_newlines(text):
    pattern = r'\n+'
    text = re.sub(pattern, '\n', text)
    text =  " ".join(text.split())
    if text[0] == "\n":
        text = text[1:]
    elif text[-1] == "\n":
        text = text[:-1]
    return text

def parse_final_prompt(prompt):
    """
    Parse the final prompt from the given prompt.

    Args:
        prompt (str): The prompt.

    Returns:
        str: The parsed final prompt.
    """
    # Define the patterns to extract the final prompt
    patterns = [
        r'<INSTRUCT>(.*?)<ENDINSTRUCT>',
        r'<INSTRUCT>(.*?)</INSTRUCT>',
        r'<<INSERT>>(.*?)<<ENDINSERT>>'
    ]
    
    # Define noise strings to remove from the final prompt
    noise_strings = [
        "The instruction that you understood is:",
        "To generate the output, follow these steps:",
        " A possible variation of the instruction is:",
        "One possible instruction is:"
    ]
    
    # Find the first match from the patterns
    for pattern in patterns:
        matches = re.findall(pattern, prompt, flags=re.DOTALL | re.MULTILINE)
        if matches:
            final_prompt = matches[0]
            break
    else:
        print("Could not parse the prompt", prompt)
        return remove_extra_newlines(prompt)

    # Remove noise strings from the final prompt
    for noise_string in noise_strings:
        final_prompt = final_prompt.replace(noise_string, "")
    return final_prompt#remove_extra_newlines(final_prompt)   

def generate_gpt4_response(system_prompt, user_prompt, gpt_model, args):
    prompt_gen_instruct = llm_utils.get_gpt4_template(system_prompt, user_prompt)
    # Generate response using LLM
    gpt_response, usage = llm.gpt4(prompt_gen_instruct, model=gpt_model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)[1:]
    # Process the response
    processed_response = parse_final_prompt(gpt_response)
    return processed_response, usage

def parse_prompt(prompt):
    lines = prompt.split('\n')
    structured_prompt = {}
    current_h1 = None
    current_h2 = None
    current_h3 = None

    for line in lines:
        h1_match = re.match(r"# (.+)", line)
        h2_match = re.match(r"## (.+)", line)
        h3_match = re.match(r"### (.+)", line)
        item_match = re.match(r"(\d+\.|\*|\-) (.+)", line)
        
        if h1_match:
            current_h1 = h1_match.group(1).strip()
            structured_prompt[current_h1] = {'body': ''}
            current_h2 = None
            current_h3 = None
        elif h2_match:
            current_h2 = h2_match.group(1).strip()
            if current_h1:
                structured_prompt[current_h1][current_h2] = {'body': ''}
            current_h3 = None
        elif h3_match:
            current_h3 = h3_match.group(1).strip()
            if current_h1 and current_h2:
                structured_prompt[current_h1][current_h2][current_h3] = {'body': ''}
        elif item_match:
            item_text = item_match.group(2).strip()
            item_key = item_match.group(1).strip()
            if current_h1 and current_h2 and current_h3:
                structured_prompt[current_h1][current_h2][current_h3][item_key] = item_text
            elif current_h1 and current_h2:
                structured_prompt[current_h1][current_h2][item_key] = item_text
            elif current_h1:
                structured_prompt[current_h1][item_key] = item_text
        else:
            body_text = line.strip()
            if body_text:
                if current_h1 and current_h2 and current_h3:
                    structured_prompt[current_h1][current_h2][current_h3]['body'] += body_text + " "
                elif current_h1 and current_h2:
                    structured_prompt[current_h1][current_h2]['body'] += body_text + " "
                elif current_h1:
                    structured_prompt[current_h1]['body'] += body_text + " "

    return structured_prompt
                