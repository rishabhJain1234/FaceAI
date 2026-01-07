"""
Simplified Face Service - Client-Side Inference Version

This service only handles embedding matching.
All heavy ML work (detection + recognition) is done on the client side.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

class FaceService:
    """
    Lightweight face matching service
    No TensorFlow, no ONNX Runtime, no OpenCV needed!
    """
    
    @staticmethod
    def validate_embedding(embedding: List[float]) -> bool:
        """
        Validate embedding format
        """
        if not embedding or len(embedding) != 512:
            return False
        
        # Check if all values are valid floats
        try:
            arr = np.array(embedding, dtype=np.float32)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False
            return True
        except:
            return False
    
    @staticmethod
    def normalize_embedding(embedding: List[float]) -> np.ndarray:
        """
        L2 normalize embedding vector
        """
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        # Both should already be normalized, but double-check
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0:
            emb1 = emb1 / norm1
        if norm2 > 0:
            emb2 = emb2 / norm2
        
        return float(np.dot(emb1, emb2))
    
    @staticmethod
    def match_embedding(
        query_embedding: List[float],
        registered_students: List[Dict],
        threshold: float = 0.45
    ) -> Tuple[Optional[Dict], float]:
        """
        Match a single embedding against registered students
        
        Args:
            query_embedding: 512-d embedding from client
            registered_students: List of student documents with embeddings
            threshold: Similarity threshold for matching
            
        Returns:
            (matched_student, similarity_score) or (None, best_score)
        """
        # Validate and normalize query embedding
        if not FaceService.validate_embedding(query_embedding):
            return None, 0.0
        
        query_emb = FaceService.normalize_embedding(query_embedding)
        
        best_score = 0.0
        best_student = None
        
        for student in registered_students:
            # Get student embedding
            student_embedding = student.get('face_embedding')
            if not student_embedding:
                continue
            
            # Validate and normalize
            if not FaceService.validate_embedding(student_embedding):
                continue
            
            student_emb = FaceService.normalize_embedding(student_embedding)
            
            # Calculate similarity
            similarity = FaceService.cosine_similarity(query_emb, student_emb)
            
            if similarity > best_score:
                best_score = similarity
                best_student = student
        
        # Apply threshold
        if best_score >= threshold:
            return best_student, best_score
        
        return None, best_score
    
    @staticmethod
    def match_multiple_embeddings(
        embeddings: List[Dict],  # [{"vector": [...], "bbox": [...], "score": ...}, ...]
        registered_students: List[Dict],
        threshold: float = 0.45
    ) -> Dict:
        """
        Match multiple embeddings (from attendance image) against registered students
        
        Args:
            embeddings: List of embedding dicts from client
            registered_students: List of student documents
            threshold: Similarity threshold
            
        Returns:
            {
                "present_students": [...],
                "absent_students": [...],
                "unknown_count": int,
                "total_detected": int,
                "matches": [...]  # Debug info
            }
        """
        present_students = []
        present_ids = set()
        unknown_count = 0
        matches = []
        
        for emb_data in embeddings:
            vector = emb_data.get('vector')
            if not vector:
                continue
            
            # Match this embedding
            matched_student, score = FaceService.match_embedding(
                vector,
                registered_students,
                threshold
            )
            
            if matched_student:
                student_id = matched_student['student_id']
                
                # Avoid duplicates (same student detected multiple times)
                if student_id not in present_ids:
                    present_students.append(matched_student)
                    present_ids.add(student_id)
                
                matches.append({
                    "student_id": student_id,
                    "name": matched_student['name'],
                    "score": round(score, 3),
                    "bbox": emb_data.get('bbox', []),
                    "quality": emb_data.get('quality', 100),
                    "thumbnail": emb_data.get('thumbnail')
                })
            else:
                unknown_count += 1
                matches.append({
                    "student_id": None,
                    "name": "Unknown",
                    "score": round(score, 3),
                    "bbox": emb_data.get('bbox', []),
                    "quality": emb_data.get('quality', 100),
                    "thumbnail": emb_data.get('thumbnail')
                })
        
        # Determine absent students
        all_student_ids = {s['student_id'] for s in registered_students}
        absent_ids = all_student_ids - present_ids
        absent_students = [s for s in registered_students if s['student_id'] in absent_ids]
        
        return {
            "present_students": present_students,
            "absent_students": absent_students,
            "unknown_count": unknown_count,
            "total_detected": len(embeddings),
            "matches": matches
        }
