import requests
import time
OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3:8b",  # Make sure this model is pulled in your Ollama instance
    "prompt": "Write in brief about the Taj Mahal",
    "stream": False     # Set to True if you want streamed output
}
start=time.time()
response = requests.post(OLLAMA_URL, json=payload)
end=time.time()

if response.status_code == 200:
    result = response.json()
    print(result["response"])
    print(end-start)
else:
    print(f"Error: {response.status_code} - {response.text}")
