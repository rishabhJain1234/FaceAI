from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from database import students_collection
from services.face_service_client import FaceService
from services.face_processor import face_processor_service
import uuid
from datetime import datetime
import numpy as np

router = APIRouter()

# Deprecated: Client-side embedding model
class StudentCreate(BaseModel):
    name: str
    student_id: str
    face_embedding: List[float]

@router.post("/add")
async def add_student(
    file: UploadFile = File(...),
    name: str = Form(...),
    student_id: str = Form(...)
):
    """
    Add student with server-side processing
    """
    # Check if student already exists
    existing = students_collection.find_one({"student_id": student_id})
    if existing:
        raise HTTPException(400, f"Student {student_id} already exists")
    
    # Check if name already exists
    existing_name = students_collection.find_one({"name": name})
    if existing_name:
        raise HTTPException(400, f"Student with name '{name}' already exists")

    try:
        # Read file content
        image_content = await file.read()
        
        # Process image to get embedding
        # We expect exactly one face for registration
        results = face_processor_service.process_image(image_content)
        
        if len(results) == 0:
             raise HTTPException(400, "No face detected in the image")
        
        if len(results) > 1:
             raise HTTPException(400, "Multiple faces detected. Please upload an image with a single face.")
             
        embedding = results[0]['vector']
        
        # Validate embedding
        if not FaceService.validate_embedding(embedding):
            raise HTTPException(400, "Invalid embedding generated")
            
        # Normalize embedding
        normalized_emb = FaceService.normalize_embedding(embedding)
        
        # Store in database
        student_data = {
            "student_id": student_id,
            "name": name,
            "face_embedding": normalized_emb.tolist(),
            "created_at": datetime.utcnow()
        }
        
        students_collection.insert_one(student_data)
        
        return {
            "message": "Student added successfully",
            "student_id": student_id,
            "name": name
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error adding student: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

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
