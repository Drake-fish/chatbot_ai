from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
load_dotenv()

chat = ChatOpenAI()

memory = ConversationBufferMemory(
  memory_key="messages", 
  return_messages=True,
  chat_memory=FileChatMessageHistory("messages.json")
)

prompt = ChatPromptTemplate(
  input_variables = ["content", "messages "],
  messages = [
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(
  llm=chat,
  prompt=prompt,
  memory=memory
)

while True:
  content = input(">>")

  result = chain({"content": content})
  
  print(result["text"])
