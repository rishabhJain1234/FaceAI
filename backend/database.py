from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "attendance_db")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

students_collection = db["students"]
attendance_collection = db["attendance"]

def get_db():
    return db
