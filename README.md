# Using GPT-4o Model with Azure OpenAI

This guide covers how to use the GPT-4o model with Azure OpenAI services.

## Prerequisites
- An Azure account with access to OpenAI.
- Basic knowledge of REST API calls.

## Steps to Get Started
1. **Set Up Azure OpenAI:**  
   - Log in to your Azure portal and create a new Azure OpenAI resource.
   - Note your subscription key and endpoint.

2. **Install Required Libraries:**  
   If you are using Python, install the `requests` library:
   ```bash
   pip install requests

3. **Making API Calls:**
Use the following Python code to interact with the GPT-4o model:

```Python
import requests

url = '<YOUR_AZURE_OPENAI_ENDPOINT>/openai/deployments/<YOUR_MODEL>/completions?api-version=2023-05-15'
headers = {
    'Content-Type': 'application/json',
    'api-key': '<YOUR_API_KEY>'
}
data = {
    'prompt': 'Your prompt here',
    'max_tokens': 150
}
response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## Understanding Parameters:

`prompt`: The text you want the model to complete.

`max_tokens`: The maximum number of tokens to generate.

## Review the Output:
Examine the output of the model and adjust your prompts as necessary for desired results.

## Conclusion
The GPT-4o model offers advanced capabilities for various applications, from conversational agents to content generation. Experiment with different prompts to get the best results!

