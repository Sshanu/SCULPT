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

    def inference(self, ex, prompt, task, index=-1, base_index_list_index=0):
        """promptProc = self.template.replace('???INSERT SYSTEM PROMPT HERE???', prompt)
        promptProc = promptProc.replace('???INSERT USER PROMPT HERE???', ex['text'])
        llm.gpt4(promptProc, temperature=0, top_p=1, model=self.opt['evaluator-engine'], max_tokens=1024)"""
        
        gpt_response = utils.invoke_llm(self.opt['evaluator_engine'], prompt, ex['text'], temperature=0, top_p=1, max_tokens=1024, index=index, base_index_list_index=base_index_list_index)
        
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
        
        if task == "bbh-formal-fallacies":
            if 'INVALID' in response.strip().upper():
                pred=1
            elif 'VALID' in response.strip().upper():
                pred = 0
            else:
                pred = 0
        elif task == "bbh-causal-judgement":
            if 'YES' in response.strip().upper():
                pred = 1
            elif 'NO' in response.strip().upper():
                pred = 0
            else:
                pred = 0
                
        elif task == "go_emotions":
            pred = go_emotion_answer_extraction(response)   

        elif task == "beaver_tails":
            pred = beaver_tails_answer_extraction(response)

        elif task == "bbh-disambiguation-qa":
            if "(a)" in response or "(A)" in response:
                pred = 0
            elif "(b)" in response or "(B)" in response:
                pred = 1
            elif "(c)" in response or "(C)" in response:
                pred = 2
            elif "(d)" in response or "(D)" in response:
                pred = 3
            else:
                pred=-1
        elif task == "bbh-salient-translation":
            if "(a)" in response or "(A)" in response:
                pred = 0
            elif "(b)" in response or "(B)" in response:
                pred = 1
            elif "(c)" in response or "(C)" in response:
                pred = 2
            elif "(d)" in response or "(D)" in response:
                pred = 3
            elif "(e)" in response or "(E)" in response:
                pred = 4
            elif "(f)" in response or "(F)" in response:
                pred = 5
            else:
                pred = -1
        else:
            raise ValueError(f"Unknown task: {task}")
        return pred, (response, gpt_response[1]), token_usage
