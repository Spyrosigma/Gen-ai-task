from config import GROQ_API_KEY
from groq import Groq


class LLMProvider:

    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY,
        )

    def get_summary(self, text: str):
        chat_completion = self.client.chat.completions.create(
            messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF assistant designed to summarize document content. Provide clear, concise summaries based on the information provided. If the content is too long, break it down into manageable parts. Try to keep the summary SHORT ! "
            },
            {
                "role": "user",
                "content": text,
            }
            ],
            model="llama-3.3-70b-versatile",
        )

        return(chat_completion.choices[0].message.content)

    def query(self, query: str):
        chat_completion = self.client.chat.completions.create(
            messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF assistant designed to answer questions about document content. Provide clear, concise responses based on the information provided. If the answer isn't in the content, acknowledge that and don't make up information. For complex topics, break down your explanation into digestible parts."
            },
            {
                "role": "user",
                "content": query,
            }
            ],
            model="llama-3.3-70b-versatile",
        )

        return(chat_completion.choices[0].message.content)
    

