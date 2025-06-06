import requests
import json

# List of base URLs for the OpenAI instances (ports)
base_urls1 = ["http://localhost:11430/v1", "http://localhost:11431/v1", "http://localhost:11432/v1", "http://localhost:11433/v1", 
              "http://localhost:11434/v1", "http://localhost:11435/v1", "http://localhost:11436/v1", "http://localhost:11437/v1", 
              "http://localhost:11440/v1", "http://localhost:11441/v1", "http://localhost:11442/v1", "http://localhost:11443/v1", 
              "http://localhost:11444/v1", "http://localhost:11445/v1", "http://localhost:11446/v1", "http://localhost:11447/v1"]

base_urls2 = ["http://localhost:11420/v1", "http://localhost:11421/v1", "http://localhost:11422/v1", "http://localhost:11423/v1", 
              "http://localhost:11424/v1", "http://localhost:11425/v1", "http://localhost:11426/v1", "http://localhost:11427/v1", 
              "http://localhost:11410/v1", "http://localhost:11411/v1", "http://localhost:11412/v1", "http://localhost:11413/v1", 
              "http://localhost:11414/v1", "http://localhost:11415/v1", "http://localhost:11416/v1", "http://localhost:11417/v1", 
              "http://localhost:11450/v1", "http://localhost:11451/v1", "http://localhost:11452/v1", "http://localhost:11453/v1", 
              "http://localhost:11454/v1", "http://localhost:11455/v1", "http://localhost:11456/v1", "http://localhost:11457/v1"]

# Combine the base URLs
base_urls = base_urls1 + base_urls2

# Define the data that will be sent in the POST request
def create_request_payload(model_name, system_prompt, user_prompt):
    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": user_prompt,
        "template": ""
    }
    return payload

# Function to send POST request to each port with model and prompts
def send_requests_to_ports(model_name, system_prompt, user_prompt):
    for base_url in base_urls:
        try:
            # Construct the full API URL for /api/chat
            api_url = base_url.replace("/v1", "") + "/api/chat"
            
            # Prepare the request data
            data = create_request_payload(model_name, system_prompt, user_prompt)
            
            # Send the POST request
            response = requests.post(api_url, json=data)
            
            # Check if the response is OK (HTTP status code 200)
            if response.status_code == 200:
                print(f"Service running on {base_url}: Response - {response.json()}")
            else:
                print(f"Service on {base_url} returned status code {response.status_code}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            # Print error message with details if there is a connection issue
            print(f"Could not connect to service on {base_url}. Error: {e}")

# Example usage:
model_name = "llama3.1"  # Replace with the model you want to use
system_prompt = "This is the system prompt."  # Replace with your system prompt
user_prompt = "Hello"  # Replace with your user message

# Send the requests to all ports
send_requests_to_ports(model_name, system_prompt, user_prompt)
