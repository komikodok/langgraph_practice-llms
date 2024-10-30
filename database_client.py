from pymongo import MongoClient
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

host = os.getenv("MONGODB_URI")

client = MongoClient(host=host, serverSelectionTimeoutMS=5000)
db = client.get_database("langgraph_practice")

collection_documents = db.get_collection("documents")
collection_chat_history = db.get_collection("chat_history")

def get_db_connection():
    return db


if __name__ == "__main__":
    print(find_dotenv())
    try:
        databases = client.list_database_names()
        print("Connected to MongoDB Atlas! Databases:", databases)
    except Exception as e:
        print("Connection error:", e)