from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from database import students_collection, attendance_collection
from services.face_service_client import FaceService
from datetime import datetime
import uuid

router = APIRouter()

class EmbeddingData(BaseModel):
    vector: List[float]
    bbox: List[float]
    score: float
    quality: float = 100.0
    thumbnail: Optional[str] = None

class AttendanceRequest(BaseModel):
    embeddings: List[EmbeddingData]

@router.post("/mark")
async def mark_attendance(request: AttendanceRequest):
    """
    Mark attendance using pre-computed embeddings from client
    """
    try:
        # Get all registered students
        students = list(students_collection.find({}))
        if not students:
            raise HTTPException(status_code=400, detail="No students registered yet")
            
        # Match embeddings
        # The client sends a list of embedding objects. We transform them for the service.
        input_embeddings = [
            {
                "vector": e.vector, 
                "bbox": e.bbox, 
                "score": e.score, 
                "quality": e.quality,
                "thumbnail": e.thumbnail
            } 
            for e in request.embeddings
        ]
        
        result = FaceService.match_multiple_embeddings(
            input_embeddings,
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
        # Note: result['present_students'] already contain student docs
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
            "matches": result.get('matches', []) # Optional debug info
        }
        
        attendance_record = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "summary": response_data
        }
        attendance_collection.insert_one(attendance_record)
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error marking attendance: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
