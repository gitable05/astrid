from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path


template = """ You are Astrid, an AI chatbot. You will help in analyzing and summarizing the research articles stored in
               vectorized database. 

               Here is the context: {context}

               Here are the results of the similarity search from the database: {results}

               Here is the user input: {user_input}   
           """

model = OllamaLLM(model="llama3.2:latest")
embedding = OllamaEmbeddings(model="mxbai-embed-large")
CHROMA_PATH = "chroma"
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    
    db = Chroma(collection_name="references",
                persist_directory=CHROMA_PATH,
                embedding_function=embedding)

    context = ""
    search_results = ""
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "/exit":
            break            

        results = db.similarity_search_with_score(user_input, k=5)
        context += "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        response = chain.invoke({"context":context,"results":results,"user_input":user_input})
        sources = [{"Article": doc.metadata.get("title", "No title"), "Text":doc.page_content} for doc, _score in results]
        formatted_response = f"Astrid: {response}\n\n Sources: \n {sources}"

        print(formatted_response)
        print("(Type '/exit' to exit)")

        context += f"\nUser : {user_input} \nAI: {response}"


if __name__ == "__main__":

    print("Hi. What can I do for you today? Type '/exit' to exit.")
    handle_conversation()
    

