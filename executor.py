import io
import traceback
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from llm import generate_code, repair_code, interpret_result, extract_python_code

EXEC_TIMEOUT = 15
MAX_ATTEMPTS = 3  # 1 original + 2 retries


def _execute_code_worker(code: str, queue: mp.Queue):
    import io
    import contextlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": __builtins__})

        img_bytes = None
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            plt.close("all")

        queue.put({
            "success": True,
            "output": output_buffer.getvalue(),
            "image_bytes": img_bytes,
            "error": None
        })

    except Exception:
        plt.close("all")
        queue.put({
            "success": False,
            "output": output_buffer.getvalue(),
            "image_bytes": None,
            "error": traceback.format_exc()
        })


def execute_code_with_timeout(code: str, timeout: int = EXEC_TIMEOUT):
    queue = mp.Queue()
    process = mp.Process(target=_execute_code_worker, args=(code, queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "success": False,
            "output": "",
            "image_bytes": None,
            "error": f"Execution timed out after {timeout} seconds."
        }

    if queue.empty():
        return {
            "success": False,
            "output": "",
            "image_bytes": None,
            "error": "Execution failed without returning any result."
        }

    return queue.get()


def run_agent(prompt: str):
    raw_code = generate_code(prompt)
    code = extract_python_code(raw_code)

    attempt = 0
    last_error = None

    while attempt < MAX_ATTEMPTS:
        result = execute_code_with_timeout(code, EXEC_TIMEOUT)

        if result["success"]:
            if attempt == 0:
                status = "Executed on first try"
            else:
                status = f"Fixed and executed on retry (attempt {attempt})"
            break

        last_error = result["error"]
        attempt += 1

        if attempt >= MAX_ATTEMPTS:
            return (
                code,
                f"Execution error (after {attempt-1} retries): {last_error}",
                "Retry failed",
                f"The system attempted to fix the code multiple times but failed. Final error: {last_error}",
                None
            )

        code = repair_code(prompt, code, last_error)

    img = None
    if result["image_bytes"] is not None:
        img = Image.open(io.BytesIO(result["image_bytes"]))

    execution_output = result["output"]

    if not execution_output.strip() and img is None:
        execution_output = "Code executed successfully, but nothing was printed."
    elif not execution_output.strip():
        execution_output = "Plot generated successfully."

    interpretation = interpret_result(prompt, code, execution_output, status)

    return code, execution_output, status, interpretation, img