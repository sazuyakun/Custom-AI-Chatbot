
# Custom AI Chatbot

The chatbot is designed to answer questions based on a specific context, utilizing Langchain for document embedding and retrieval and Flask RESTful APIs to handle conversations.

Make sure to have Ollama installed and running in your local device with "llama3". If not installed, then use this [link](https://ollama.com/) to download it and run the following command in your favorite terminal.
```
ollama pull llama3
```
## Getting Started
Note: Make sure to have python installed in your device to use this. Download [link](https://www.python.org/downloads/)
1. Clone the repository:
```
git clone https://github.com/sazuyakun/Custom-AI-Chatbot
```
2. Navigate into the project directory:
```
cd Custom-AI-Chatbot/
```
3. Create a new virtual environment:
```
python -m venv venv
. venv/bin/activate
```
4. Install the requirements:
```
pip install -r requirements.txt
```
5. Run the app:
```
python Chat-bot.py
```
You should be able to access the app at http://localhost:5000

