import re
import os
import json
import copy
import llm
import ollama
from collections import Counter
import dirtyjson
import traceback

class PromptMetadata:

    def __init__(self, prompt, index, gentype):
        self.prompt = prompt
        self.index = index
        self.generation_type = gentype

def invoke_llm(model, system_prompt, user_prompt, max_tokens, temperature, top_p, index=-1):
    if model == "gpt4o":
        template = "<|im_start|>system\n???INSERT SYSTEM PROMPT HERE???\n<|im_end|>\n<|im_start|>user\n???INSERT USER PROMPT HERE???\n<|im_end|>\n<|im_start|>assistant"
        prompt = template.replace('???INSERT SYSTEM PROMPT HERE???', system_prompt)
        prompt = prompt.replace('???INSERT USER PROMPT HERE???', user_prompt)
        return llm.gpt4(prompt, model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    else:
        return ollama.ollama(user_prompt, system_prompt, model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, index=index)

def parse_prompt(prompt):
    lines = prompt.split('\n')
    structured_prompt = {}
    heading_stack = []
    numeric_counter = 1
    in_numeric_list = False

    def add_body_text(heading_stack, text):
        current_dict = structured_prompt
        for heading in heading_stack:
            current_dict = current_dict.setdefault(heading, {})
        if current_dict.get('body', '') != '':
            current_dict['body'] = current_dict.get('body', '') + '\n' + text.strip() + " "
        else:
            current_dict['body'] = current_dict.get('body', '') + text.strip() + " "
        current_dict['Examples'] = []
        
    def add_example_text(heading_stack, examples, item_key):
        current_dict = structured_prompt
        for heading in heading_stack:
            current_dict = current_dict.setdefault(heading, {})
        if item_key is not None:
            item_dict = current_dict.setdefault(item_key, {})
            item_dict['Examples'] = examples
        else:
            for example in examples:
                current_dict.setdefault('Examples', []).append(example)

    def add_item_text(heading_stack, item_text, item_key):
        current_dict = structured_prompt
        for heading in heading_stack:
            current_dict = current_dict.setdefault(heading, {})
        if 'body' not in current_dict:
            current_dict.setdefault('body', "")
        if 'Examples' not in current_dict:
            current_dict.setdefault('Examples', [])            
        item_dict = current_dict.setdefault(item_key, {})
        item_dict['body'] = item_text
        item_dict.setdefault('Examples', [])
        

    def handle_numeric_or_bulleted_item(item_key):
        nonlocal numeric_counter, in_numeric_list
        if item_key in ('*', '-'):
            if not in_numeric_list:
                numeric_counter = 1
                in_numeric_list = True
            item_key = str(numeric_counter) + '.'
            numeric_counter += 1
        else:
            in_numeric_list = False
        return item_key
    
    previous_item_match = None
    for line in lines:
        h_match = re.match(r"(#+) (.+)", line)
        item_match = re.match(r"(\d+|\*|\-) (.+)", line)
        if h_match:
            level = len(h_match.group(1))
            heading = h_match.group(2).strip()
            
            # Adjust the stack to the correct heading level
            if len(heading_stack) < level:
                heading_stack.append(heading)
            else:
                heading_stack = heading_stack[:level-1]
                heading_stack.append(heading)
            # Reset numeric list status when heading changes
            in_numeric_list = False
            previous_item_match = None
        elif item_match:
            item_text = item_match.group(2).strip()
            item_key = item_match.group(1).strip()
            item_key = handle_numeric_or_bulleted_item(item_key)
            add_item_text(heading_stack, item_text, item_key)
            previous_item_match = item_key
        elif "Examples:" in line:
            example_text = [ex.strip() for ex in re.findall(r'\{([^}]+)\}', line)]
            add_example_text(heading_stack, example_text, previous_item_match)
        else:
            body_text = line.strip()
            if body_text:
                add_body_text(heading_stack, body_text)

    return structured_prompt

def deep_update(mapping: dict[str, any], keys: list[str], i : int, value: any):
    updated_mapping = mapping.copy()
    key = keys[i].strip()
    if i < len(keys) - 1:
        updated_mapping[key] = deep_update(updated_mapping[key], keys, i+1, value)
    else:
        if isinstance(updated_mapping, str):
            raise Exception("Something went wrong")
        elif isinstance(value, str) or isinstance(value, list):
            updated_mapping[key] = value
        elif isinstance(value, dict):
            if not value:
                updated_mapping[key] = value
            else:
                for k, v in value.items():
                    updated_mapping[key][k] = v
    return updated_mapping

def convert_attributeddict_to_dict(structured_prompt: dict[str, any]):
    converted_dict = {}
    for key, value in structured_prompt.items():
        key = key.strip()
        if isinstance(value, dict):
            converted_dict[key] = convert_attributeddict_to_dict(value)
        else:
            converted_dict[key] = value
    return converted_dict

def contains_number(string):
    return any(char.isdigit() for char in string)

def dict_to_prompt(parsed_prompt: dict[str, any], depth: int, prompt: str):
    for key, value in parsed_prompt.items():
        key = key.strip()
        empty_value = False
        if isinstance(value, dict):
            for sub_keys, sub_value in value.items():
                if isinstance(sub_value, dict) and not sub_value:
                    empty_value = True
                elif isinstance(sub_value, list) and not sub_value:
                    empty_value = True
                elif isinstance(sub_value, str) and sub_value == "":
                    empty_value = True
                else:
                    empty_value = False
                    break
        else:
            empty_value = False
        if empty_value:
            continue
        if isinstance(value, dict) and contains_number(key.replace(".", "")):
            prompt += '* '
            prompt = dict_to_prompt(value, depth, prompt) 
        elif isinstance(value, dict):
            if prompt != "":
                prompt += '\n'
            prompt += '#' * depth + ' ' + key + '\n'
            prompt = dict_to_prompt(value, depth + 1, prompt)             
        elif key == "body":
            if value != "":
                prompt += value.strip() + '\n'
        elif key == "Examples" and len(value) > 0:
            prompt += "Examples: "
            for i, item in enumerate(value):
                prompt += '{' + item + '}'
                if i < len(value) - 1:
                    prompt += ", "
            prompt += '\n'
        elif isinstance(value, str):
            prompt += '* ' + value + '\n'
    return prompt

def deep_copy_keys(mapping: dict[str, any], depth: int, copied_mapping: dict[str, any]):
    for key, value in mapping.items():
        key = key.strip()
        if isinstance(value, dict):
            copied_mapping[key] = {}
            copied_mapping[key] = deep_copy_keys(value, depth + 1, copied_mapping[key])
        else:
            copied_mapping[key] = "..."
    return copied_mapping

def expand_prompt(parsed_prompt: dict[str, any], redacted_prompt: dict[str, any], keys: list[str], i: int):
    updated_prompt = redacted_prompt.copy()
    key = keys[i].strip()
    if i < len(keys) - 1:
        updated_prompt[key] = expand_prompt(parsed_prompt[key], redacted_prompt[key], keys, i+1)
    else:
        if key in parsed_prompt.keys():
            updated_prompt[key] = parsed_prompt[key]
        else:
            #if exact match is not found, check if it exists as substring in any key / value in the prompt
            for k, v in parsed_prompt.items():
                if k in key or key in k:
                    updated_prompt[k] = v
                    break
                elif str(v).startswith(key):
                    updated_prompt[k] = v
                    break
    return updated_prompt
    
def get_value_at_level(mapping: dict[str, any], keys: list[str], level: int):
    key = keys[level].strip()  
    if level == len(keys) - 1:
        if isinstance(mapping, dict):
            if key in mapping.keys():
                return mapping[key]
            else:
                #if exact match is not found, check if it exists as substring in any key / value in the prompt
                for k, v in mapping.items():
                    if k in key or key in k:
                        return v
                    elif str(v).startswith(key):
                        return v
            return []
        else:
            raise Exception("Invalid key")
    return get_value_at_level(mapping[key], keys, level+1)

def get_dict_at_level(mapping: dict[str, any], keys: list[str], level: int):
    key = keys[level].strip()
    if level == len(keys) - 1:
        if key in mapping.keys():
            return mapping
        else:
            #if exact match is not found, check if it exists as substring in any key / value in the prompt
            for k, v in mapping.items():
                k_substring = key.split("-")[1]
                if k in key or key in k:
                    return mapping
                elif k_substring in k:
                    return mapping
                elif str(v).startswith(key):
                    return mapping
    return get_dict_at_level(mapping[key], keys, level+1)

def update_token_usage(token_usage, token_details):
    for key in token_details.keys():
        if key in token_usage.keys():
            token_usage[key] += token_details[key]
        else:
            token_usage[key] = token_details[key]
    return token_usage

def find_headers_in_reference(references):
    found_headers = []
    for reference in references:
        keys = reference.split(">")
        found_headers.append(keys[0].strip())
    return found_headers

def find_and_correct_reference(reference, data_structure):
    try:
        corrected_reference = ""
        keys = reference.split('>')
        current = data_structure
        if len(keys)==1:
            if keys[0] in current:
                return reference
        found = True
        
        # Check if the full reference exists
        for key in keys:
            key = key.strip()
            if key in current:
                current = current[key]
            else:
                # If a key is missing, try to find the closest match
                closest_matches = []
                for parent_key in current.keys():
                    if key in current[parent_key]:
                        closest_matches.append(f"{parent_key}> {key}")

                if closest_matches:
                    # Add all closest matches to corrected_references
                    for match in closest_matches:
                        corrected_reference = match
                    print(f"Corrected reference: {reference} to {closest_matches}")
                    found = False
                else:
                    print(f"Missing key in reference: {reference}")
                break
        
        if found:
            corrected_reference = reference
            
    except Exception as e:
        print(f"Error {e} while correct_references {reference}")
        print(traceback.format_exc())                
        corrected_reference = reference
    return corrected_reference
    
def find_and_correct_references(references, data_structure):
    corrected_references = []
    for ref in references:
        corrected_references.append(find_and_correct_reference(ref, data_structure))
    return corrected_references

def copy_prompt(prompt):
    prompt = copy.deepcopy(json.loads(json.dumps(prompt, indent=4)))
    return prompt

def clean_empty_keys(content):
    if isinstance(content, dict):
        cleaned_dict = {}
        for key, value in content.items():
            # If the key is 'body' or 'Examples', retain it regardless of being empty
            if key in ['body', 'Examples']:
                cleaned_dict[key] = value
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                nested_clean = clean_empty_keys(value)
                if nested_clean:
                    cleaned_dict[key] = nested_clean
            elif isinstance(value, list):
                # Retain lists that are not empty
                if value:
                    cleaned_dict[key] = value
            elif value:
                # Retain values that are not empty
                cleaned_dict[key] = value
        return cleaned_dict
    return content

def read_action_response(file_path):
    action_json = None
    with open(file_path, 'r') as file:
        action_str = file.read() 
        json_part = action_str.split("```")[0].strip()
        try:
            action_json = dirtyjson.loads(json_part)  
        except Exception as e:
            print(f"Error {e} decoding JSON in file: {file_path}")
    return  action_json
                    
def extract_action_types(directory_path):
    action_types = []
    for filename in os.listdir(directory_path):
        if 'actor_response' in filename or 'exampleupdate_response' in filename:  
            file_path = os.path.join(directory_path, filename)
            action_json = read_action_response(file_path)
            if action_json:
                if 'actor_response' in filename:
                    for action in action_json.get('actions', []):
                        action_type = action.get('action_type')
                        if action_type != "Example Update":  
                            action_types.append(action_type)
                else:
                    action_types.append("Example Update - " + action_json['update_type'])
                        

    # Count the frequency of each action type
    return Counter(action_types)

def correct_section_position(reference):
    if reference[-1] == ">":
        reference = reference[:-1]
    keys = reference.split(">")
    keys = [key for key in keys if key not in ['body', 'Examples']]
    return keys

def check_keys_with_period(section):
    keys_with_period = [key for key in section if '.' in key]
    return keys_with_period
