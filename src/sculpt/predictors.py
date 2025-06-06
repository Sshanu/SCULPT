from abc import ABC, abstractmethod
import utils
import re
import json

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

    @abstractmethod
    def inference(self, ex, prompt):
        pass

def get_score_from_xlm(response, key):
    match = re.search(r'<{}>\s*(-?\d+\.?\d*)\s*</{}>'.format(key, key), response, re.IGNORECASE)
    # If no match, try to handle the JSON-like format inside the tags (both list and dictionary forms)
    if not match:
        # Match a JSON-like structure with a generic key-value pair inside the tag
        match = re.search(r'<{}>\s*\{{.*?(-?\d+\.?\d*).*\}}\s*</{}>'.format(key, key), response, re.IGNORECASE)
    
    if match:
        return float(match.group(1))  # Return as float to handle both integers and floats
    else:
        print("error", response, key)
    return -1

def go_emotion_answer_extraction(response):
    response = response.strip()
    
    # Remove unwanted text patterns that might wrap the numbers
    response = re.sub(r'<\|.*?\|>', '', response)  # Removes tags like <|imadas_sep|>
    response = re.sub(r'<end>', '', response)  # Removes <end> if present
    response = response.strip()

    # Extract numbers from the cleaned response
    numbers = re.findall(r'\d+', response)
    
    # If numbers exist, return them as a properly formatted comma-separated string
    return ",".join(numbers) if numbers else "-1"

def beaver_tails_answer_extraction(response):
    response = re.sub(r'\s+', ' ', response).strip()
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return response
        json_string = match.group(0).replace('\n', '').replace('\r', '')
        return json_string
    except json.JSONDecodeError:
        return response

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']
    template = "<|im_start|>system\n???INSERT SYSTEM PROMPT HERE???\n<|im_end|>\n<|im_start|>user\n???INSERT USER PROMPT HERE???\n<|im_end|>\n<|im_start|>assistant"

    def inference(self, ex, prompt, task, index=-1):
        """promptProc = self.template.replace('???INSERT SYSTEM PROMPT HERE???', prompt)
        promptProc = promptProc.replace('???INSERT USER PROMPT HERE???', ex['text'])
        llm.gpt4(promptProc, temperature=0, top_p=1, model=self.opt['evaluator-engine'], max_tokens=1024)"""
        
        gpt_response = utils.invoke_llm(self.opt['evaluator_engine'], prompt, ex['text'], temperature=0, top_p=1, max_tokens=1024, index=index)
        response = gpt_response[1]
        token_usage = gpt_response[2]
        
        if "could not get a response" in response.lower():
            return 0, response, token_usage
        
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        if "<|fim_suffix|>" in response:
            response = response.split("<|fim_suffix|>")[0]
        if "<|im_start|>" in response:
            response = response.split("<|im_start|>")[1]
        
        if task == "sug-defensive":
            if '1' in response.strip().upper():
                pred = 1
            elif '0' in response.strip().upper():
                pred = 0
            else:
                pred = -1
                                    
        elif task == "bbh-formal-fallacies":
            if 'INVALID' in response.strip().upper():
                pred=1
            elif 'VALID' in response.strip().upper():
                pred = 0
            else:
                pred = -1
                
        elif task == "bbh-causal-judgement":
            if 'YES' in response.strip().upper():
                pred = 1
            elif 'NO' in response.strip().upper():
                pred = 0
            else:
                pred = -1

        elif task == "go_emotions":
            pred = go_emotion_answer_extraction(response)   

        elif task == "beaver_tails":
            pred = beaver_tails_answer_extraction(response)
        
        elif task == "bbh-disambiguation-qa":
            if "(A)" in response:
                pred = 0
            elif "(B)" in response:
                pred = 1
            elif "(C)" in response:
                pred = 2
            elif "(D)" in response:
                pred = 3
            else:
                pred = -1
    
        elif task == "bbh-salient-translation":
            if "(A)" in response:
                pred = 0
            elif "(B)" in response:
                pred = 1
            elif "(C)" in response:
                pred = 2
            elif "(D)" in response:
                pred = 3
            elif "(E)" in response:
                pred = 4
            elif "(F)" in response:
                pred = 5
            else:
                pred = -1
        else:
            raise ValueError(f"Unknown task: {task}")
        return pred, response, token_usage