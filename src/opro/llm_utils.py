import os
def get_gpt4_template(sys_prompt, user_prompt):
    """
    Get the template for the prompt.

    Args:
        sys_prompt (str): The system prompt.
        user_prompt (str): The user prompt.

    Returns:
        str: The prompt template.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, 'general_gpt4_prompt_template.txt'), "r") as file:
        prompt = file.read()
    return prompt.replace("???INSERT SYSTEM PROMPT HERE???",sys_prompt).replace("???INSERT USER PROMPT HERE???", user_prompt)

def get_slm_template(sys_prompt, user_prompts, model_name, tokenizer):
    """
    Get the template for SLM.

    Args:
        sys_prompt (str): The system prompt.
        user_prompts (list): List of user prompts.
        slm (str): The SLM model to use.

    Returns:
        list: List of chat prompts.
    """
    chat_prompts = []
    for user_prompt in user_prompts:
        if "zephyr" == model_name:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
            ) + "<|assistant|>"
            chat_prompts.append(prompt)
        elif "mistral" == model_name:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": (sys_prompt + "\n" + user_prompt)}],
                tokenize=False,
            )
            chat_prompts.append(prompt)
        elif model_name in ["phi3", "llama3"]:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
            )
            chat_prompts.append(prompt)
    return chat_prompts

def get_slm_batch_size(slm, task):
    """
    Get the batch size for SLM.

    Args:
        slm (str): The SLM model.
        task (str): The task.

    Returns:
        int: The batch size.
    """
    # Adjust batch size values based on the SLM and task
    if "phi" in slm:
        batch_size = 2
    else:
        batch_size = 1
    
    print(f"SLM: {slm}, Task: {task}, New Batch Size: {batch_size}")  # Print the SLM, task, and new batch size
    return batch_size    


def get_gpt4_infer_llm_for_metric_score_hyperparams(model, max_new_tokens, num_worker):
    """
    Get the hyperparameters for GPT-4 inference for metric scoring.

    Args:
        max_new_tokens (int): The maximum number of new tokens.
        num_worker (int): The number of workers.

    Returns:
        dict: The hyperparameters.
    """
    return {
        "model": model,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 0.01,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "retry": 20,
        "num_worker": num_worker
    }

def get_slm_infer_llm_for_metric_score_hyperparams(max_new_tokens):
    """
    Get the hyperparameters for SLM inference for metric scoring.

    Args:
        max_new_tokens (int): The maximum number of new tokens.

    Returns:
        dict: The hyperparameters.
    """
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False
    }