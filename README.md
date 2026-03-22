# Using GPT-4o Model with Azure OpenAI

This guide covers how to use the GPT-4o model with Azure OpenAI services.

## **Overview**

GPT-4o is the latest multimodal model from OpenAI that can handle both text and images simultaneously. This notebook demonstrates how to use it with Azure OpenAI services.

## Prerequisites
- An Azure account with access to OpenAI.
- Basic knowledge of REST API calls.

---

## **1. Installation & Setup**

```python
# Install required packages
pip install openai requests gradio python-dotenv pillow
```

---

## **2. Import Required Libraries**

```python
import base64
import datetime
import glob
import gradio as gr
import openai
import os
import requests
import sys
from dotenv import load_dotenv
from io import BytesIO
from mimetypes import guess_type
from openai import AzureOpenAI
from PIL import Image
```

---

## **3. Version Check**

```python
def check_openai_version():
    """
    Check Azure OpenAI version
    """
    installed_version = openai.__version__

    try:
        version_number = float(installed_version[:3])
    except ValueError:
        print("Invalid OpenAI version format")
        return

    print(f"Installed OpenAI version: {installed_version}")

    if version_number < 1.0:
        print("⚠️  Warning: Upgrade OpenAI to version >= 1.0.0")
        print("To upgrade, run: pip install openai --upgrade")
    else:
        print(f"✅ OpenAI version {installed_version} is >= 1.0.0")

# Call the function
check_openai_version()
```

---

## **4. Configuration & Authentication**

```python
from dotenv import load_dotenv

# Load environment variables from azure.env file
load_dotenv("azure.env")

# Azure OpenAI Configuration
api_type = "azure"
api_key = os.getenv("OPENAI_API_KEY")  # Your Azure API key
api_base = os.getenv("OPENAI_API_BASE")  # Your Azure endpoint
api_version = "2024-05-01-preview"  # API version

# Model name (deployed in Azure OpenAI Studio)
model = "gpt-4o"
```

---

## **5. Text-Only Queries**

### **Function to Query GPT-4o with Text**

```python
def gpt4o_text(prompt):
    """
    Send a text prompt to GPT-4o and get a response
    """
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{model}",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.7,
    )

    return response
```

### **Example Usage: Text Queries**

```python
# Example 1: Simple introduction
result = gpt4o_text("Who are you?")
print(result.choices[0].message.content)
# Output: I am an AI assistant created to help you with information...

# Example 2: General knowledge
result = gpt4o_text("What is the capital of France?")
print(result.choices[0].message.content)
# Output: The capital of France is **Paris**.

# Example 3: Complex explanation
result = gpt4o_text("What is an ARMA model?")
print(result.choices[0].message.content)
# Output: An ARMA model (AutoRegressive Moving Average model)...
```

---

## **6. Image Analysis Using URL**

### **Function to Analyze Images from URL**

```python
def gpt4o_url(image_url, prompt):
    """
    Send an image URL and prompt to GPT-4o
    """
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{model}",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to analyse images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )

    return response
```

### **Example: Analyze Image from URL**

```python
import requests
from PIL import Image
from io import BytesIO

# Image URL
image_url = "https://raw.githubusercontent.com/retkowsky/images/master/jo.png"

# Fetch and display the image
response = requests.get(image_url)
print("Status:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))

if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img.show()

# Analyze the image
result = gpt4o_url(image_url, "Analyse this image and describe what you see")
print(result.choices[0].message.content)
# Output: This image shows the Arc de Triomphe in Paris illuminated at night...
```

---

## **7. Image Analysis Using Local Files**

### **Helper Function: Convert Local Image to Data URL**

```python
def local_image_to_data_url(image_path):
    """
    Convert a local image file to a base64 data URL
    """
    mime_type, _ = guess_type(image_path)

    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"
```

### **Function to Analyze Local Image Files**

```python
def gpt4o_imagefile(image_file, prompt):
    """
    Send a local image file and prompt to GPT-4o
    """
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{model}",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to analyse images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": local_image_to_data_url(image_file)},
                    },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )

    return response
```

### **Example: Analyze Local Image**

```python
# Analyze a local image file
image_file = "image1.jpg"
result = gpt4o_imagefile(image_file, "What do you see in this image?")
print(result.choices[0].message.content)
```

---

## **8. Complete Example Workflow**

```python
# Step 1: Check versions
check_openai_version()

# Step 2: Text query
print("--- Text Query ---")
text_response = gpt4o_text("Explain machine learning in simple terms")
print(text_response.choices[0].message.content)

# Step 3: Image URL analysis
print("\n--- Image URL Analysis ---")
image_url = "https://example.com/image.png"
image_response = gpt4o_url(image_url, "What objects are in this image?")
print(image_response.choices[0].message.content)

# Step 4: Local image analysis
print("\n--- Local Image Analysis ---")
local_response = gpt4o_imagefile("local_image.jpg", "Describe this image")
print(local_response.choices[0].message.content)
```

---

## **Key Parameters Explained**

| Parameter | Description |
|-----------|-------------|
| `api_key` | Your Azure OpenAI API key |
| `api_base` | Your Azure OpenAI endpoint URL |
| `api_version` | API version (e.g., `2024-05-01-preview`) |
| `model` | Deployed model name (e.g., `gpt-4o`) |
| `max_tokens` | Maximum response length (1-2000) |
| `temperature` | Randomness (0.0 = deterministic, 1.0 = creative) |

---

## **Documentation Reference**

📚 [Azure OpenAI GPT-4o Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4o-and-gpt-4-turbo)

This guide covers all the essential aspects of using GPT-4o with Azure OpenAI!


## Conclusion
The GPT-4o model offers advanced capabilities for various applications, from conversational agents to content generation. Experiment with different prompts to get the best results!

