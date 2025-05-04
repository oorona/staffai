import requests

def chat_with_model(token):
    url = 'http://llm:3000/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "seniordev",
      "messages": [
        {
          "role": "system", 
          "content": "You can use web search to answer questions if needed.",
          "role": "user",
          "content": "What are the top AI news today?"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


print(chat_with_model("sk-40499a5ecb23447dbccc19f3890fecab"))