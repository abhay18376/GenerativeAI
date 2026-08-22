# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage

# # Initialize model (replace with your Google API key)
# chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyAzGLpk7BWX3dUPA_wtt5cnlCX5yghR6Os")

# response = chat.invoke([HumanMessage(content="Hello, what is LangChain?")])
# print(response.content)


from langchain_google_genai import ChatGoogleGenerativeAI
print("import OK:", ChatGoogleGenerativeAI)
