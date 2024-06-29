from flask import Flask, request, jsonify
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

embeddings = OllamaEmbeddings(
    model="llama3"
)

vector = FAISS.from_documents(docs, embeddings)

llm = Ollama(
    model="llama3"
)

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

app = Flask(__name__)
api = Api(app)

class Chatbot(Resource):
    def post(self):
        # Parse the input data
        data = request.get_json()
        question = data.get('question')
        
        # Get the response from the retrieval chain
        response = retrieval_chain.invoke({
            "input": question
        })
        
        return jsonify({'response': response})
    
api.add_resource(Chatbot, '/chatbot')

if __name__ == '__main__':
    app.run(debug=True)