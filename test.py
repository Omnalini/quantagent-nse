from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key="AIzaSyA4ANaQkrGqtcqwFtCG1fN5UJgvyQdZRok",
)

response = llm.invoke("Say hello in one sentence.")
print(response.content)