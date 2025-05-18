import ollama
import time

start=time.time()
response = ollama.chat(
model="llama3:8b",
messages=[{"role": "user", "content": "Write in brief about the Taj Mahal"}] )
print(response["message"]["content"])
end=time.time()
ft=end-start
print("Time taken for execution: "+str(ft))