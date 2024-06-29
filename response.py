import requests

url = 'http://127.0.0.1:5000/chatbot'
data = {'question': 'What technical courses are available?'}
response = requests.post(url, json=data)

print(response.json())
