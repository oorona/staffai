import os
from openai import OpenAI          # new import style (v1.x)
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(os.getenv("OPENAI_API_KEY"))
client = OpenAI(   api_key=os.getenv("OPENAI_API_KEY") )

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

personalities = [
    "Albert Einstein",
    "William Shakespeare",
    "Yoda, the Jedi Master"
]

question = "Tell me a very short explanation why the sky is blue."

for persona in personalities:
    response = client.chat.completions.create(          # new call path
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are {persona}. Reply in the well‑known voice and style of {persona}."
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=60,
    )

    print(f"\n— {persona} —")
    print(response.choices[0].message.content.strip())

