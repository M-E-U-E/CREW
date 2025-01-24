# import requests
# from dotenv import load_dotenv
# import os
# import time

# # Load environment variables
# load_dotenv()

# # Securely retrieve API key
# GROC_API_KEY = os.getenv("GROC_API_KEY")
# if not GROC_API_KEY:
#     raise ValueError("GROC API Key not found. Please set it in the .env file.")

# # Define Base URL for GROC API
# BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# class GrocAgent:
#     """Base class for interacting with GROC API."""
    
#     def __init__(self, api_key: str, base_url: str = BASE_URL, model: str = "llama-3.3-70b-versatile"):
#         self.api_key = api_key
#         self.base_url = base_url
#         self.model = model

#     def _send_request(self, messages, max_tokens=300, temperature=0.7):
#         """Send a request to the GROC API."""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#         }
#         try:
#             response = requests.post(self.base_url, json=payload, headers=headers)
#             response.raise_for_status()
#             # Extract the first response choice
#             return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
#         except requests.exceptions.RequestException as e:
#             print(f"Error interacting with GROC API: {e}")
#             return None


# class SummaryAgent(GrocAgent):
#     """Agent responsible for summarizing content."""
    
#     def summarize(self, content: str):
#         """Summarize the given content."""
#         messages = [
#             {"role": "system", "content": "You are an AI assistant that summarizes content."},
#             {"role": "user", "content": f"Summarize the following content: {content}"},
#         ]
#         return self._send_request(messages)


# class ContentFetcher:
#     """Utility class to fetch content from a URL."""
    
#     @staticmethod
#     def fetch(url: str):
#         """Fetch content from the given URL."""
#         try:
#             response = requests.get(url)
#             response.raise_for_status()  # Raise HTTPError for bad responses
#             return response.text
#         except requests.exceptions.RequestException as e:
#             print(f"Error fetching content from URL: {e}")
#             return None


# # Main workflow
# def main():
#     # URL to fetch content from
#     url = "https://documentation-using-ai-agent.readthedocs.io/en/latest/overview/"
    
#     # Fetch the content
#     content = ContentFetcher.fetch(url)
#     if content:
#         print("Content fetched successfully.\n")
        
#         # Create an instance of the SummaryAgent
#         summary_agent = SummaryAgent(api_key=GROC_API_KEY)
        
#         # Step 1: Summarize content
#         summary = summary_agent.summarize(content)
#         if summary:
#             print("Summary:\n", summary)
#             time.sleep(2)
            
#             # Step 2: Answer a question based on the summary (COMMENTED OUT)
#             # question = "What is the purpose of the documentation?"
#             # answer_agent = AnswerAgent(api_key=GROC_API_KEY)
#             # answer = answer_agent.answer_question(summary, question)
#             # if answer:
#             #     print("\nAnswer to the question:\n", answer)
#             # else:
#             #     print("Failed to get an answer from the Answer Agent.")
#         else:
#             print("Failed to summarize content with the Summary Agent.")
#     else:
#         print("Failed to fetch content from the provided URL.")


# if __name__ == "__main__":
#     main()
# # 
import requests
from dotenv import load_dotenv
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

# Load environment variables
load_dotenv()

# Set up logging for better debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Securely retrieve API key
GROC_API_KEY = os.getenv("GROC_API_KEY")
if not GROC_API_KEY:
    raise ValueError("GROC API Key not found. Please set it in the .env file.")

@dataclass
class CrewAIAgent:
    """
    Representation of a Crew AI Agent with specific attributes.
    Defines the agent's purpose, role, and tools for operation.
    """
    name: str = "Content Insight Extractor"
    role: str = "Content Intelligence Specialist"
    goal: str = "Extract and distill key insights from web-based documentation"
    backstory: str = (
        "An advanced information processing agent designed to transform "
        "lengthy web content into digestible, actionable summaries."
    )
    tools: List[str] = field(default_factory=lambda: [
        "Web Content Retrieval", 
        "AI-Powered Summarization", 
        "API-Driven Processing"
    ])

class BaseAgent:
    """
    Base class for interacting with APIs, providing a flexible and reusable structure.
    Handles request preparation and response parsing.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _send_request(self, messages: List[dict], max_tokens: int = 300, temperature: float = 0.7) -> Optional[str]:
        """
        Send a request to the API with the given messages and parameters.
        Returns the processed response content or None on failure.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            logging.info("Sending request to the API...")
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
            logging.info("Response received successfully.")
            return content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error interacting with the API: {e}")
            return None

class SummaryAgent(BaseAgent):
    """
    Specialized agent for summarizing content with role-specific context.
    Uses a CrewAIAgent profile to enhance request messages.
    """

    def __init__(self, api_key: str, agent_profile: CrewAIAgent):
        super().__init__(api_key, base_url="https://api.groq.com/openai/v1/chat/completions", model="llama-3.3-70b-versatile")
        self.agent_profile = agent_profile

    def summarize(self, content: str) -> Optional[str]:
        """
        Summarize the provided content based on the agent's contextual understanding.
        """
        system_context = (
            f"You are {self.agent_profile.name}, a {self.agent_profile.role}. "
            f"Your goal is to {self.agent_profile.goal}. "
            f"Backstory: {self.agent_profile.backstory}"
        )
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": f"Analyze and summarize the following content, focusing on key insights: {content}"},
        ]
        return self._send_request(messages)

class ContentFetcher:
    """
    Utility class to handle fetching content from a URL.
    """

    @staticmethod
    def fetch(url: str) -> Optional[str]:
        """
        Fetch and return the content from the given URL.
        Returns None if an error occurs.
        """
        try:
            logging.info(f"Fetching content from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            logging.info("Content fetched successfully.")
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching content from URL: {e}")
            return None

def main():
    """
    Main function to orchestrate the process:
    - Fetch content from a URL.
    - Use CrewAIAgent and SummaryAgent to process the content.
    - Output the summarized content.
    """
    # Create the Crew AI Agent profile
    content_agent = CrewAIAgent()
    
    # URL to fetch content from
    url = "https://documentation-using-ai-agent.readthedocs.io/en/latest/features/"
    
    # Fetch the content
    content = ContentFetcher.fetch(url)
    if content:
        # Initialize the SummaryAgent with the CrewAIAgent profile
        summary_agent = SummaryAgent(api_key=GROC_API_KEY, agent_profile=content_agent)
        
        # Summarize the fetched content
        summary = summary_agent.summarize(content)
        if summary:
            logging.info("Summary successfully generated.")
            print("Summary:\n", summary)
        else:
            logging.warning("Failed to generate a summary.")
    else:
        logging.warning("Failed to fetch content from the provided URL.")

if __name__ == "__main__":
    main()
