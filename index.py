from flask import Flask, render_template, request, jsonify
import os
from apikey import apikey
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from chatbot import setup_chain

os.environ['OPENAI_API_KEY'] = apikey
chain = setup_chain()

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Get the chat message from the POST data
        message = request.form.get('message')
        
        # Pass the chat message to the LLM chain
        response = chain.run(message)

        # Return the chatbot's response as JSON
        return jsonify({'message': message, 'bot_reply': response})

    else:
        return render_template('chat.html')

if __name__ == "__main__":
    app.run(debug=True)
