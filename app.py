import gradio as gr
import requests
import os
import io
import contextlib

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_code(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are a Python code generator.

STRICT RULES:
- Output ONLY raw Python code
- No explanations
- No markdown
- No text before or after the code
- Print important results clearly

Use pandas, matplotlib, yfinance if needed.
"""

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    return result["choices"][0]["message"]["content"]

def run_agent(prompt):
    code = generate_code(prompt)
    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {})
        execution_output = output_buffer.getvalue()
        if not execution_output.strip():
            execution_output = "Code executed successfully, but nothing was printed."
    except Exception as e:
        execution_output = f"Execution error: {str(e)}"

    return code, execution_output

demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(label="Prompt", lines=2),
    outputs=[
        gr.Code(label="Generated Python Code", language="python"),
        gr.Textbox(label="Execution Output", lines=12)
    ],
    title="AI Data Analysis Agent"
)

demo.launch()