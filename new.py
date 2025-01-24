import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Securely retrieve API key
GROC_API_KEY = os.getenv("GROC_API_KEY")
if not GROC_API_KEY:
    raise ValueError("GROC API Key not found. Please set it in the .env file.")

# Define Base URL for GROC API
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to fetch content from a URL
def fetch_content_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {e}")
        return None

# Function to interact with GROC API for chat completions
def groc_chat_completion(messages):
    headers = {
        "Authorization": f"Bearer {GROC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",  # Replace with the correct model as per GROC documentation
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
    }
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers)
        print(response.text)  # Debug response
        response.raise_for_status()
        # Extract the first response choice
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
    except requests.exceptions.RequestException as e:
        print(f"Error interacting with GROC API: {e}")
        return None

# URL to fetch content from
url = "https://documentation-using-ai-agent.readthedocs.io/en/latest/"

# Fetch content from the URL
content = fetch_content_from_url(url)
if content:
    # Prepare conversation messages
    messages = [
        {"role": "system", "content": "You are an AI assistant helping summarize and answer questions."},
        {"role": "user", "content": f"Summarize the following content: {content}"},
    ]
    
    # Step 1: Summarize content
    summary = groc_chat_completion(messages)
    if summary:
        print("\nSummary:")
        print(summary)

        # Step 2: Answer a question based on the summary
        messages.append({"role": "user", "content": f"Based on the summary, what is the purpose of the documentation?"})
        answer = groc_chat_completion(messages)
        if answer:
            print("\nAnswer to the question:")
            print(answer)
        else:
            print("Failed to get an answer from GROC API.")
    else:
        print("Failed to summarize content with GROC API.")
else:
    print("Failed to fetch content from the provided URL.")
