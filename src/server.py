# libraries
import json
from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
from dotenv import load_dotenv

import os
import glob
import pprint
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import logging
# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level (optional)
logger.setLevel(logging.DEBUG)

# Create a handler for outputting logs to stdout
stdout_handler = logging.StreamHandler()

# Set the logging level for the handler (optional)
stdout_handler.setLevel(logging.INFO)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the handler
stdout_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stdout_handler)



load_dotenv()  # load env vars from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"openai.api_key={openai.api_key}")

chromadb_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection_ncc2022 = chromadb_client.get_collection(name="ncc2022", embedding_function=sentence_transformer_ef) 



logger.info(f"collections {chromadb_client.list_collections()}")
logger.info(f"number of entries: {collection_ncc2022.count()}")
# text = collection_ncc2022.query(query_texts=["What's the maximum travel distance between two fire exits in a shopping center?"],
#                      include=['documents', 'distances', 'metadatas'],
#                      n_results=3)
# logger.info(text)

app = Flask(__name__)
# run_with_ngrok(app) 

@app.route("/")
def index():
    return render_template("index_server.html")


@app.route("/get_response")
def get_response():
    question = request.args.get("message")
    texts = collection_ncc2022.query(query_texts=[question],
                     include=['documents', 'distances', 'metadatas'],
                     n_results=3
                     )
    logger.info(f"retrieved texts:{texts}")
    messages = build_prompt_with_context(question, texts['documents'][0])
    logger.info(f"openai.api_key={openai.api_key}")
    response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=1000,
        )
    answer = response['choices'][0].get("message").get("content")
    
    return answer

def build_prompt_with_context(question, context):
    system_prompt = {
        "role": "system",
        "content": """You are a helpful assistant who knows a lot about architecture standards in Australia. 
        I will ask you questions about architecture standards. I will provide a number of relevant paragraphs extracted from 
        the architecture standards for your reference."""
    }
    user_prompt = {
        "role": "user",
        "content": f"""
        Please answer the question delimited by ``` regarding archituecture standards in Australia. The texts delimited by === are 
        a number of paragraphs extracted from National Construction Code. You can use the texts as reference. Please provide your reply in format of json with 3 fields: "answer",
        "confidence score" between 0 to 1, and "source" which are the original clauses supporting your answer. 
        
        Question:```
        {question}
        ```
        Texts:===
        {os.linesep.join(context)}
        ===
        """
    }
    return [system_prompt,
            user_prompt]



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)