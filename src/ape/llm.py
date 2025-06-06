import re
import openai
import sys
import os

def convert_prompt_to_messages(prompt):
    """
    Converts a raw prompt with <|im_start|>...<|im_end|> format into OpenAI Chat API messages.
    """
    messages = []

    # Extract system message
    system_match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if system_match:
        messages.append({"role": "system", "content": system_match.group(1).strip()})

    # Extract all user-assistant message pairs
    conversation_matches = re.findall(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
        prompt, re.DOTALL
    )
    for user, assistant in conversation_matches:
        messages.append({"role": "user", "content": user.strip()})
        messages.append({"role": "assistant", "content": assistant.strip()})

    # Extract unmatched final user message
    last_user_matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if len(last_user_matches) > len(conversation_matches):
        messages.append({"role": "user", "content": last_user_matches[-1].strip()})

    return messages

def gpt4(prompt, model="gpt-4o", max_tokens=256, temperature=0, top_p=1.0, frequency_penalty=0, presence_penalty=0, stop_words=None, retry=5):
    """
    Sends a structured prompt in <|im_start|> format to the OpenAI API using chat messages.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    messages = convert_prompt_to_messages(prompt)

    for i in range(retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_words
            )
            response_text = response.choices[0].message.content
            usage = response.usage
            return prompt, (response_text, usage)
        except Exception as e:
            print(f"API call failed (attempt {i+1}/{retry+1}): {e}", file=sys.stderr)
            if i == retry:
                print("Retry Count exceeded. Could not get a response.", file=sys.stderr)
                return prompt, ("Could not get a response", {})