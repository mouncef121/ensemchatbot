from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='./')

# Load the knowledge base from a JSON file
def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

# Save the knowledge base to a JSON file
def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Find the best match for the user's question using semantic similarity
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    user_question_embedding = model.encode([user_question])
    question_embeddings = model.encode(questions)
    similarities = cosine_similarity(user_question_embedding, question_embeddings)[0]
    best_match_index = similarities.argmax()
    if similarities[best_match_index] > 0.6:
        return questions[best_match_index]
    else:
        return None

# Get the answer for the given question
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

# Main function to implement the chat bot
def chat_bot(user_input: str):
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')
    if user_input.lower() == 'quit':
        return "Goodbye!"
    best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])
    if best_match:
        answer: str = get_answer_for_question(best_match, knowledge_base)
        return answer
    else:
        return "I don't know the answer. Can you teach me?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = chat_bot(user_input)
    return jsonify({'response': response})
    
if __name__ == '__main__':
    app.run(debug=True)
