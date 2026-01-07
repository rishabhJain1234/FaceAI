from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from database import students_collection
from services.face_service_client import FaceService
import uuid
from datetime import datetime

router = APIRouter()

class StudentCreate(BaseModel):
    name: str
    student_id: str
    face_embedding: List[float]

@router.post("/add")
async def add_student(student: StudentCreate):
    """
    Add student with pre-computed embedding from client
    """
    # Validate embedding
    if not FaceService.validate_embedding(student.face_embedding):
        raise HTTPException(400, "Invalid embedding format (expected 512-d vector)")
    
    # Check if student already exists
    existing = students_collection.find_one({"student_id": student.student_id})
    if existing:
        raise HTTPException(400, f"Student {student.student_id} already exists")
    
    # Check if name already exists
    existing_name = students_collection.find_one({"name": student.name})
    if existing_name:
        raise HTTPException(400, f"Student with name '{student.name}' already exists")
    
    try:
        # Normalize embedding
        normalized_emb = FaceService.normalize_embedding(student.face_embedding)
        
        # Store in database
        student_data = {
            "student_id": student.student_id,
            "name": student.name,
            "face_embedding": normalized_emb.tolist(),
            "created_at": datetime.utcnow()
        }
        
        students_collection.insert_one(student_data)
        
        return {
            "message": "Student added successfully",
            "student_id": student.student_id,
            "name": student.name
        }
    except Exception as e:
        print(f"Error adding student: {e}")
        raise HTTPException(500, "Internal server error")

@router.get("/list")
async def list_students():
    """
    Get all students (without embeddings for efficiency)
    """
    students = list(students_collection.find({}, {"_id": 0, "face_embedding": 0}))
    return students

@router.get("/")
async def get_all_students():
    """
    Get all students with embeddings (for attendance matching)
    """
    students = list(students_collection.find({}, {"_id": 0}))
    return students
