import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_response(prompt, model="gpt-3.5-turbo"):
    """Get response from OpenAI's LLM."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return None

def analyze_code(code):
    """Analyze code using OpenAI's API."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code analysis expert. Analyze the provided code and identify potential issues, bugs, and areas for improvement."},
                {"role": "user", "content": f"Please analyze this code:\n\n{code}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error analyzing code: {e}")
        return None

def generate_test_cases(code):
    """Generate test cases for the provided code."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a testing expert. Generate comprehensive test cases for the provided code."},
                {"role": "user", "content": f"Please generate test cases for this code:\n\n{code}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating test cases: {e}")
        return None

def optimize_code(code):
    """Optimize the provided code."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code optimization expert. Optimize the provided code for better performance and readability."},
                {"role": "user", "content": f"Please optimize this code:\n\n{code}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error optimizing code: {e}")
        return None

def document_code(code):
    """Generate documentation for the provided code."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a documentation expert. Generate comprehensive documentation for the provided code."},
                {"role": "user", "content": f"Please generate documentation for this code:\n\n{code}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating documentation: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint to analyze code."""
    try:
        data = request.get_json()
        code = data.get('code')
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Get analysis from OpenAI
        analysis = analyze_code(code)
        
        if not analysis:
            return jsonify({"error": "Failed to analyze code"}), 500
        
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test():
    """Endpoint to generate test cases."""
    try:
        data = request.get_json()
        code = data.get('code')
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Generate test cases
        test_cases = generate_test_cases(code)
        
        if not test_cases:
            return jsonify({"error": "Failed to generate test cases"}), 500
        
        return jsonify({"test_cases": test_cases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """Endpoint to optimize code."""
    try:
        data = request.get_json()
        code = data.get('code')
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Optimize code
        optimized_code = optimize_code(code)
        
        if not optimized_code:
            return jsonify({"error": "Failed to optimize code"}), 500
        
        return jsonify({"optimized_code": optimized_code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/document', methods=['POST'])
def document():
    """Endpoint to generate documentation."""
    try:
        data = request.get_json()
        code = data.get('code')
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Generate documentation
        documentation = document_code(code)
        
        if not documentation:
            return jsonify({"error": "Failed to generate documentation"}), 500
        
        return jsonify({"documentation": documentation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 