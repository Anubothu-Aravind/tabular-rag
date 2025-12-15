import requests
import json

def llm(prompt: str) -> str:
    """
    Connect to local Ollama instance
    """
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response from LLM")
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve'"
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"