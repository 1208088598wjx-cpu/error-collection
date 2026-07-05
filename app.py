import os
import json
import openai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route("/api/solve", methods=["POST"])
def solve():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "缺少题目"}), 400
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "题目不能为空"}), 400

    system_prompt = (
        "You are a mathematics solution expert. Analyze the given problem independently and dynamically. "
        "Provide a structured solution. Output MUST be a JSON object with the following keys:\n"
        "  - solution: step-by-step solution (as a string, use \\n for newlines)\n"
        "  - visual: description of any helpful diagram (string)\n"
        "  - name: topic name (string)\n"
        "  - condition: core prerequisites/conditions (string)\n"
        "  - contrast: confusable concepts or similar problems (string)\n"
        "  - signal: keywords that trigger this method (string)\n"
        "  - same: a practice question of the same type (string)\n"
        "  - variant: a variant practice question (string)\n"
        "  - comprehensive: a comprehensive practice question (string)\n"
        "  - chinese_summary: a concise Chinese explanation of the solution and key points (string)\n"
        "Output ONLY the JSON object, no additional text."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {question}"}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        raw_content = response.choices[0].message.content
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_content)
        for field in ["solution", "visual", "name", "condition", "contrast", "signal", "same", "variant", "comprehensive", "chinese_summary"]:
            if field not in parsed:
                parsed[field] = ""
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)