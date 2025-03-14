from openai import OpenAI
import requests
from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import re
from werkzeug.utils import secure_filename
import base64
from PyPDF2 import PdfReader
from PIL import Image
import io
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from notion_client import Client

# Configure logging
logging.basicConfig(
    handlers=[RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Load environment variables
load_dotenv()

# Flask Setup
app = Flask(__name__)

# Set API Keys and Rates
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=anthropic_api_key)
notion = Client(auth=os.getenv('NOTION_API_KEY'))

# Cost per 1K tokens (as of 2024)
COST_RATES = {
    'gpt4': {
        'input': 0.01,    # $0.01 per 1K input tokens
        'output': 0.03    # $0.03 per 1K output tokens
    },
    'gpt4_vision': {
        'input': 0.01,    # $0.01 per 1K input tokens
        'output': 0.03,   # $0.03 per 1K output tokens
        'image': 0.00765  # $0.00765 per image (for standard quality)
    },
    'dalle3': {
        'standard': 0.040,  # $0.040 per image (1024x1024, standard quality)
        'hd': 0.080        # $0.080 per image (1024x1024, HD quality)
    },
    'claude': {
        'input': 0.015,   # $0.015 per 1K input tokens
        'output': 0.045   # $0.045 per 1K output tokens
    },
    'perplexity': {
        'input': 0.008,   # $0.008 per 1K input tokens
        'output': 0.024   # $0.024 per 1K output tokens
    }
}

# Configure upload settings
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image: {str(e)}")
        return None

def compress_image(image_path, max_size_mb=4):
    """Compress image if it's larger than max_size_mb"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Calculate current size in MB
        current_size = os.path.getsize(image_path) / (1024 * 1024)
        
        if current_size > max_size_mb:
            # Calculate new size maintaining aspect ratio
            ratio = (max_size_mb / current_size) ** 0.5
            new_size = tuple(int(dim * ratio) for dim in img.size)
            
            # Resize and save with reduced quality
            img = img.resize(new_size, Image.LANCZOS)
            
            # Save to a buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            
            # Write back to file
            with open(image_path, 'wb') as f:
                f.write(buffer.getvalue())
                
        return True
    except Exception as e:
        logging.error(f"Error compressing image: {str(e)}")
        return False

def get_file_content(filepath):
    """Extract content from uploaded file"""
    try:
        file_ext = filepath.split('.')[-1].lower()
        
        if file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == 'pdf':
            return extract_text_from_pdf(filepath)
        elif file_ext in ['png', 'jpg', 'jpeg', 'gif']:
            if compress_image(filepath):
                return encode_image(filepath)
            return None
        else:
            return None
    except Exception as e:
        logging.error(f"Error getting file content: {str(e)}")
        return None

def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None

def calculate_cost(input_tokens, output_tokens, model_type, include_image=False):
    """Calculate cost based on token usage and model type"""
    try:
        rates = COST_RATES[model_type]
        input_cost = (input_tokens / 1000) * rates['input']
        output_cost = (output_tokens / 1000) * rates['output']
        image_cost = rates.get('image', 0) if include_image else 0
        return input_cost + output_cost + image_cost
    except KeyError:
        logging.error(f"Unknown model type: {model_type}")
        return 0

def format_text(text):
    """Format text with markdown-style formatting"""
    try:
        # Convert markdown to HTML but preserve whitespace
        text = text.replace('\n', '<br>')  # Convert newlines first
        text = text.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;')  # Preserve indentation
        
        # Handle markdown formatting
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        
        return text
    except Exception as e:
        logging.error(f"Error formatting text: {str(e)}")
        return text

def get_openai_response(question, file_path=None, generate_image=False):
    """Get response from OpenAI API"""
    try:
        # Handle DALL-E image generation
        if generate_image:
            response = client.images.generate(
                model="dall-e-3",
                prompt=question,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image_data = base64.b64encode(image_response.content).decode('utf-8')
                return {
                    'type': 'generated_image',
                    'content': f"data:image/png;base64,{image_data}",
                    'text': "Image generated successfully.",
                    'usage': {
                        'cost': COST_RATES['dalle3']['standard'],
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0
                    }
                }

        # Handle image analysis or text completion
        messages = [{"role": "user", "content": question}]
        
        if file_path and os.path.exists(file_path):
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                base64_image = encode_image(file_path)
                if base64_image:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300
                    )
                    
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    cost = calculate_cost(input_tokens, output_tokens, 'gpt4_vision', include_image=True)
                    
                    return {
                        'type': 'text',
                        'content': format_text(response.choices[0].message.content.strip()),
                        'usage': {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'cost': round(cost, 4)
                        }
                    }

        # Regular GPT-4 text completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calculate_cost(input_tokens, output_tokens, 'gpt4')
        
        return {
            'type': 'text',
            'content': format_text(response.choices[0].message.content.strip()),
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost': round(cost, 4)
            }
        }
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        return {
            'type': 'text',
            'content': format_text(f"Error fetching response from OpenAI: {str(e)}"),
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0}
        }

def get_perplexity_response(question):
    """Get response from Perplexity API"""
    try:
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "llama-3.1-sonar-huge-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Be precise and concise in your responses."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {perplexity_api_key}"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            input_tokens = response_data.get('usage', {}).get('prompt_tokens', len(question.split()) + 20)
            output_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
            cost = calculate_cost(input_tokens, output_tokens, 'perplexity')
            
            response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response from PerplexityAI")
            return {
                'content': format_text(response_text),
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': round(cost, 4)
                }
            }
        else:
            error_msg = f"Error: Received status code {response.status_code} from PerplexityAI"
            logging.error(error_msg)
            return {
                'content': format_text(error_msg),
                'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0}
            }
    except Exception as e:
        logging.error(f"Error in Perplexity API call: {str(e)}")
        return {
            'content': format_text(f"Error fetching response from PerplexityAI: {str(e)}"),
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0}
        }

def get_claude_response(question, file_path=None):
    """Get response from Claude API"""
    try:
        content = question
        if file_path and os.path.exists(file_path):
            file_content = get_file_content(file_path)
            if file_content:
                content = f"File content: {file_content}\n\nQuestion: {question}"

        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            temperature=0.7,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(input_tokens, output_tokens, 'claude')
        
        return {
            'content': format_text(response.content[0].text),
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost': round(cost, 4)
            }
        }
    except Exception as e:
        logging.error(f"Error in Claude API call: {str(e)}")
        return {
            'content': format_text(f"Error: {str(e)}"),
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0}
        }

def get_claude_evaluation(question, responses, has_file=False):
    """Get Claude's evaluation of all AI responses"""
    try:
        if has_file:
            eval_prompt = f"""Question: {question}

OpenAI's Response:
{responses['openai']['content']}

Claude's Response:
{responses['claude']['content']}

Please evaluate these responses carefully and determine which one is the best. Consider factors like:
1. Accuracy and relevance
2. Completeness of the answer
3. Clarity and explanation quality
4. Practical usefulness

Provide your evaluation in this format:
- Best Response: [Name of AI]
- Explanation: [Detailed comparison explaining why this response is better]"""
        else:
            eval_prompt = f"""Question: {question}

OpenAI's Response:
{responses['openai']['content']}

Perplexity's Response:
{responses['perplexity']['content']}

Claude's Response:
{responses['claude']['content']}

Please evaluate these responses carefully and determine which one is the best. Consider factors like:
1. Accuracy and relevance
2. Completeness of the answer
3. Clarity and explanation quality
4. Practical usefulness

Provide your evaluation in this format:
- Best Response: [Name of AI]
- Explanation: [Detailed comparison explaining why this response is better]"""

        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        evaluation = response.content[0].text
        return {
            'content': format_text(evaluation),
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'cost': round(calculate_cost(response.usage.input_tokens, response.usage.output_tokens, 'claude'), 4)
            }
        }
    except Exception as e:
        logging.error(f"Error in Claude evaluation: {str(e)}")
        return {
            'content': format_text(f"Error: {str(e)}"),
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0}
        }
def extract_best_response(evaluation_text):
    """Extract best AI and its response from the evaluation"""
    try:
        best_ai_match = re.search(r"Best Response:\s*(.*?)[\n\r]", evaluation_text)
        best_ai = best_ai_match.group(1).strip() if best_ai_match else None
        
        if "OpenAI" in best_ai:
            best_ai = "OpenAI"
        elif "Claude" in best_ai:
            best_ai = "Claude"
        elif "Perplexity" in best_ai:
            best_ai = "Perplexity"
            
        return best_ai
    except Exception as e:
        logging.error(f"Error extracting best response: {str(e)}")
        return None

def log_to_notion(question, best_answer, best_ai):
    """Log the best answer to Notion page"""
    try:
        page_id = os.getenv('NOTION_PAGE_ID')
        
        if not page_id:
            logging.error("Missing Notion Page ID")
            return False
        
        response = notion.pages.create(
            parent={"page_id": page_id},
            properties={
                "title": [{"text": {"content": question[:100]}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Question"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": question}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Best Answer"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": best_answer}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Best AI Model"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": best_ai}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Date"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}}]
                    }
                }
            ]
        )
        
        logging.info(f"Successfully created Notion page with ID: {response.id}")
        return True
    except Exception as e:
        logging.error(f"Error logging to Notion: {str(e)}")
        return False

def format_text(text):
    """Format text with markdown-style formatting"""
    try:
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        return text
    except Exception as e:
        logging.error(f"Error formatting text: {str(e)}")
        return text

def cleanup_file(file_path):
    """Clean up uploaded file"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        logging.error(f"Error cleaning up file: {str(e)}")
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route handler"""
    if request.method == 'POST':
        try:
            question = request.form.get('question', '').strip()
            action = request.form.get('action', 'ask')
            file = request.files.get('file')
            file_path = None
            file_url = None
            
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return render_template('index.html', 
                                        error="Invalid file type. Supported formats: txt, pdf, png, jpg, jpeg, gif")
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    with open(file_path, 'rb') as img_file:
                        file_url = f"data:image/{file_path.split('.')[-1]};base64,{base64.b64encode(img_file.read()).decode()}"
            
            generate_image = (action == 'generate')
            has_file = bool(file_url or file_path)
            
            responses = {
                'openai': get_openai_response(question, file_path, generate_image) if question else "",
                'claude': get_claude_response(question, file_path) if question and not generate_image else "",
                'perplexity': get_perplexity_response(question) if question and not (has_file or generate_image) else ""
            }

            evaluation = None
            if question and not generate_image and responses['openai'] and responses['claude']:
                evaluation = get_claude_evaluation(question, responses, has_file)
                
                best_ai = extract_best_response(evaluation['content'])
                if best_ai:
                    best_answer = ""
                    if best_ai == "OpenAI":
                        best_answer = responses['openai']['content']
                    elif best_ai == "Claude":
                        best_answer = responses['claude']['content']
                    elif best_ai == "Perplexity":
                        best_answer = responses['perplexity']['content']
                    
                    if best_answer:
                        log_to_notion(question, best_answer, best_ai)
            
            cleanup_file(file_path)
            
            total_cost = sum(r.get('usage', {}).get('cost', 0) for r in responses.values() if r)
            if evaluation:
                total_cost += evaluation['usage']['cost']
            
            return render_template('index.html', 
                                question=question,
                                file_url=file_url,
                                openai_result=responses['openai'],
                                perplexity_result=responses['perplexity'],
                                claude_result=responses['claude'],
                                evaluation=evaluation,
                                total_cost=round(total_cost, 4))
                                
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return render_template('index.html', 
                                error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)