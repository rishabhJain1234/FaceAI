from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from database import students_collection, attendance_collection
from services.face_service_client import FaceService
from datetime import datetime
import uuid

router = APIRouter()

@router.post("/mark")
async def mark_attendance(file: UploadFile = File(...)):
    """
    Mark attendance by uploading an image.
    Inference is performed on the server.
    """
    try:
        # Get all registered students
        students = list(students_collection.find({}))
        if not students:
            raise HTTPException(status_code=400, detail="No students registered yet")
            
        # Read image content
        image_content = await file.read()
        
        # Run Inference (Detect + Recognize)
        from services.face_processor import face_processor_service
        embeddings = face_processor_service.process_image(image_content)
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="No faces detected in the image")
        
        # Match embeddings using existing service logic
        # transform to existing structure if needed, but match_multiple_embeddings accepts dicts
        # face_processor returns list of dicts with: vector, bbox, score, quality, thumbnail
        
        result = FaceService.match_multiple_embeddings(
            embeddings,
            students,
            threshold=0.45
        )
        
        # Determine absent students info
        present_ids = {s['student_id'] for s in result['present_students']}
        absent_students = []
        for s in students:
            if s['student_id'] not in present_ids:
                absent_students.append({
                    "student_id": s['student_id'],
                    "name": s['name']
                })
        
        # Prepare response data matching the expected frontend structure
        present_clean = []
        for s in result['present_students']:
            present_clean.append({
                "student_id": s['student_id'],
                "name": s['name']
            })
            
        response_data = {
            "present_students": present_clean,
            "absent_students": absent_students,
            "unknown_count": result['unknown_count'],
            "total_detected": result['total_detected'],
            "total_registered": len(students),
            "matches": result.get('matches', [])
        }
        
        attendance_record = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "summary": response_data
        }
        attendance_collection.insert_one(attendance_record)
        
        return response_data
        
    except HTTPException as he:
        raise he
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error marking attendance: {e}")
        # In production, logging the stack trace is better
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
