import json
from openai import OpenAI
import random
import concurrent.futures

# List of base URLs for the OpenAI
base_urls_list = [["http://localhost:11430/v1","http://localhost:11431/v1", "http://localhost:11432/v1", "http://localhost:11433/v1", 
               "http://localhost:11434/v1", "http://localhost:11435/v1", "http://localhost:11436/v1", "http://localhost:11437/v1"], 
              ["http://localhost:11440/v1","http://localhost:11441/v1", "http://localhost:11442/v1", "http://localhost:11443/v1",
               "http://localhost:11444/v1", "http://localhost:11445/v1", "http://localhost:11446/v1", "http://localhost:11447/v1"],
              ["http://localhost:11420/v1","http://localhost:11421/v1", "http://localhost:11422/v1", "http://localhost:11423/v1",
               "http://localhost:11424/v1", "http://localhost:11425/v1", "http://localhost:11426/v1", "http://localhost:11427/v1"],
               ["http://localhost:11410/v1","http://localhost:11411/v1", "http://localhost:11412/v1", "http://localhost:11413/v1",
                "http://localhost:11414/v1", "http://localhost:11415/v1", "http://localhost:11416/v1", "http://localhost:11417/v1"],
              ["http://localhost:11450/v1","http://localhost:11451/v1", "http://localhost:11452/v1", "http://localhost:11453/v1",
               "http://localhost:11454/v1", "http://localhost:11455/v1", "http://localhost:11456/v1", "http://localhost:11457/v1"]]

def ollama(user_message, system_prompt, model="gpt4o", history=None, max_tokens=256, temperature=0, top_p=1.0, frequency_penalty=0, presence_penalty=0, retry=5, index=-1, base_index_list_index=0):
    if model == "llama3.1":
        model_name = "llama3.1"
    elif model == "phi3":
        model_name = "phi3:3.8b-mini-4k-instruct-fp16"
    elif model == "qwen2.5":
        model_name = "qwen2.5"
    elif model == "phi3.5":
        model_name = "phi3.5"
    elif model == "gemma2":
        model_name = "gemma2"
    elif model == "mistralv3":
        model_name = "mistral"
    else:
        raise ValueError("Invalid model name. Please provide a valid model name.")

    # If base_url is not provided, then select a random base_url from the list
    base_urls = base_urls_list[base_index_list_index]
    if index < len(base_urls) or index > len(base_urls):
        base_url = base_urls[index]
    else:
        base_url = random.choice(base_urls)
     
     # If history is None, then it is the first message of the conversation
    messages = history+[{"role": "user", "content": user_message}] if history else [
        {
            "role": "system",
            "content": "" 
        },
        {
            "role": "user",
            "content": system_prompt + "\nInput: " + user_message
        }
    ]

    flag = True
    while(flag):
        try:
            client = OpenAI(base_url=base_url, api_key="ollama")
            response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False)

            gen_response = response.choices[0].message.content
            token_details = dict(response.usage)
            flag = False
        except Exception as e:
            print(e)
            gen_response = "Could not get a response"
            token_details = {}
            token_details["prompt_tokens"] = 0
            token_details["completion_tokens"] = 0
            token_details["total_tokens"] = 0

        if retry==0:
            flag = False
        retry -= 1

    if gen_response == "Could not get a response":
        print("Retry Count exceeded. Could not get a response. Please check the API or your code.")
    return user_message, gen_response, token_details


# batch inference of ollama where prompts is a list of user messages and that will be uniformly distributed among the base_urls
def ollama_batch_inference(prompts, system_prompt, model="gpt4o", max_tokens=256, temperature=0, top_p=1.0, frequency_penalty=0, presence_penalty=0, retry=5):
    response_dict = {}
    # Split prompts into equal parts for each base_url
    num_prompts = len(prompts)
    base_urls = []
    base_urls = [base_urls+x for x in base_urls_list]
    num_base_urls = len(base_urls)
    prompts_per_base_url = num_prompts // num_base_urls
    # but last base_url will have the remaining prompts
    for i in range(num_base_urls):
        start = i * prompts_per_base_url
        end = (i+1) * prompts_per_base_url
        if i == num_base_urls-1:
            end = num_prompts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit threads
            futures = []
            for prompt in prompts[start:end]:
                futures.append(executor.submit(ollama, user_message=prompt, system_prompt=system_prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, retry=retry, index=i))

            # Extract output from completed threads
            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                response_dict[output[0]] = output[1]

    # Order the responses based on the order of prompts
    responses = []
    for prompt in prompts:
        if prompt in response_dict:
            responses.append(response_dict[prompt])
        else:
            responses.append("Could not get a response")
    return responses