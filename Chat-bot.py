import pickle
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(data)

embeddings = OllamaEmbeddings(model="llama3")

# vector = FAISS.from_documents(docs, embeddings)
# import pickle
# with open("FAISS_Store.pkl", "wb") as file:
#     pickle.dump(vector, file)
with open('FAISS_Store.pkl', 'rb') as file:
    vector = pickle.load(file)

llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)
def prompt_answer(prompt):
    response = retrieval_chain.invoke({"input": prompt})
    return response["answer"]



app = Flask(__name__)
api = Api(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

class Chat(Resource):
    def post(self):
        data = request.json
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        answer = prompt_answer(prompt)
        return jsonify({
            "answer": answer
        })

api.add_resource(Chat, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
    
