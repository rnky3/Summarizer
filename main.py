from pywebio import input, output, start_server, config
from pywebio.output import put_buttons, put_file, put_html, put_progressbar, set_progressbar, use_scope, remove
import requests
import time
import random as rand


result_bytes = None


def process_text(input_text):
    url = "http://127.0.0.1:5000/api/predict"
    headers = {"Content-Type": "application/json"}
    data = {"input": input_text}

    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to process text. Status code: {response.status_code}"}


styles = """
body {
    font-family: 'Arial', sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
}

.site-container {
    max-width: 800px;
    margin: 20px auto;
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

header {
    text-align: center;
    margin-bottom: 20px;
}

article {
    line-height: 1.6;
    color: #333;
}

h1, h2 {
    color: #333;
}

hr {
    margin-top: 20px;
    border: 0;
    border-top: 1px solid #ccc;
}

.meta {
    margin-bottom: 5px;
}
"""


@config(title="Summarybot", description="A too to generate summaries", css_style=styles)
def main():
    output.put_markdown("## Summarybot").style('text-align:center')
    output.put_markdown("Summarybot is summarization tool that employs post-ensemble to generate summaries using Natural Language Processing (NLP). Adaptable to various languages and content types, it excels in context comprehension and presents key information fluently. It can generate summaries across a spectrum of applications, from news articles to technical documents.")

    input_text = input.textarea("Enter the text article:", rows=14, required=True)

    with use_scope('loading'):
        put_progressbar('bar')
        progress = 0
        while progress < 950:
            delta = rand.randint(1, 50)
            set_progressbar('bar', progress / 1000)
            progress += delta
            time_elapsed = rand.randint(1, 5)
            time.sleep(0.1 + time_elapsed / 10)
        result = process_text(input_text)
        set_progressbar('bar', 1.0)
    
    remove('loading')

    if "error" in result:
        output.put_text(result["error"])
    else:
        output.put_markdown("## Summary:")
        output.put_markdown(result["message"])
        result_bytes = result["message"].encode("utf-8")
        filename = "summary.txt"
        put_file(filename, content=result_bytes, label="Download")


if __name__ == "__main__":
    start_server(main, port=80)
