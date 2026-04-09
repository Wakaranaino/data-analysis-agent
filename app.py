import gradio as gr
import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    return result["choices"][0]["message"]["content"]

demo = gr.Interface(
    fn=call_llm,
    inputs="text",
    outputs="text"
)

demo.launch()