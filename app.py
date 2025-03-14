# Required Libraries
from openai import OpenAI
import requests
from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Flask Setup
app = Flask(__name__)

# Set API Keys (Replace with your actual keys)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")  # Hypothetical, assuming Perplexity has an API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=anthropic_api_key)

# Function to call OpenAI API
def get_openai_response(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0
        )
        # Access the content of the first choice
        response_text = response.choices[0].message.content.strip()
        if '.' in response_text:
            return response_text[:response_text.rfind('.') + 1]
        return response_text
    except Exception as e:
        return f"Error fetching response from OpenAI: {str(e)}"

# Function to call PerplexityAI API
def get_perplexity_response(question):
    try:
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "Be precise and concise."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response from PerplexityAI")
            if '.' in response_text:
                return response_text[:response_text.rfind('.') + 1]
            return response_text
        else:
            return f"Error: Received status code {response.status_code} from PerplexityAI"
    except Exception as e:
        return f"Error fetching response from PerplexityAI: {str(e)}"

# Function to call Claude API without retries or delays
def get_claude_response(question):
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {"role": "user", "content": question}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data.get("completion", "No response from Claude")
            if '.' in response_text:
                return response_text[:response_text.rfind('.') + 1]
            return response_text
        else:
            return f"Error: Received status code {response.status_code} from Claude: {response.text}"
    except Exception as e:
        return f"Error fetching response from Claude: {str(e)}"

# Function to evaluate responses using OpenAI API with o1-preview model
def get_openai_evaluation(question, openai_result, perplexity_result, claude_result):
    evaluation_prompt = f"Here are three responses to the question: '{question}'.\n\n1. OpenAI Response: {openai_result}\n2. PerplexityAI Response: {perplexity_result}\n3. Claude Response: {claude_result}\n\nPlease evaluate the quality of these responses and state which one is the best, providing a reason why."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=300,
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        if '.' in response_text:
            return response_text[:response_text.rfind('.') + 1]
        return response_text
    except Exception as e:
        return f"Error fetching evaluation from OpenAI: {str(e)}"

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        openai_result = get_openai_response(question)
        perplexity_result = get_perplexity_response(question)
        claude_result = get_claude_response(question)
        evaluation_result = get_openai_evaluation(question, openai_result, perplexity_result, claude_result)
        return render_template('index.html', question=question, openai_result=openai_result, perplexity_result=perplexity_result, claude_result=claude_result, evaluation_result=evaluation_result)
    return render_template('index.html')

# HTML Template (index.html)
html_code = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <title>AI Response Comparison</title>
    <style>
      body {
        background-color: #f8f9fa;
      }
      .container {
        margin-top: 50px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
      }
      h1 {
        font-weight: bold;
        margin-bottom: 30px;
      }
      .form-group {
        margin-bottom: 20px;
      }
      .response-container {
        margin-top: 30px;
        padding: 15px;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        background-color: #f1f1f1;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .response-title {
        font-weight: bold;
        margin-top: 20px;
      }
      .response-text {
        font-size: 1.1em;
        line-height: 1.6;
        text-align: left;
      }
      .highlight {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
      }
    </style>
  </head>
  <body>
    <div class="container text-center">
      <h1>Multi-AI Response Tool</h1>
      <form method="post">
        <div class="form-group">
          <input type="text" name="question" class="form-control" placeholder="Ask a question" required>
        </div>
        <button type="submit" class="btn btn-primary">Get Answers</button>
      </form>
      {% if question %}
      <div class="response-container mt-4">
        <h3>Question: {{ question }}</h3>
        <div class="response-title">OpenAI Response:</div>
        <p class="response-text">{{ openai_result|safe }}</p>
        <div class="response-title">PerplexityAI Response:</div>
        <p class="response-text">{{ perplexity_result|safe }}</p>
        <div class="response-title">Claude Response:</div>
        <p class="response-text">{{ claude_result|safe }}</p>
        <div class="response-title">OpenAI Evaluation:</div>
        <p class="response-text">{{ evaluation_result|safe }}</p>
        
        {% if 'OpenAI' in evaluation_result %}
        <div class="response-title">Chosen Response:</div>
        <p class="response-text highlight">OpenAI Response: {{ openai_result|safe }}</p>
        {% elif 'PerplexityAI' in evaluation_result %}
        <div class="response-title">Chosen Response:</div>
        <p class="response-text highlight">PerplexityAI Response: {{ perplexity_result|safe }}</p>
        {% elif 'Claude' in evaluation_result %}
        <div class="response-title">Chosen Response:</div>
        <p class="response-text highlight">Claude Response: {{ claude_result|safe }}</p>
        {% endif %}
      </div>
      {% endif %}
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

# Write the HTML template to a file (for simplicity)
with open('templates/index.html', 'w') as file:
    file.write(html_code)

# Running the Flask App
if __name__ == '__main__':
    app.run(debug=True)
