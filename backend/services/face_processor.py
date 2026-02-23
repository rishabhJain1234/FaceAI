import numpy as np
import onnxruntime as ort
import cv2
import os
from typing import List, Dict, Tuple, Optional

class FaceProcessor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.det_session = None
        self.rec_session = None
        
        # SCRFD Parameters
        self.det_input_size = (640, 640)
        self.det_threshold = 0.5
        self.nms_threshold = 0.5
        
        # MobileFaceNet Parameters
        self.rec_input_size = (112, 112)
        
    def load_models(self):
        """Initialize ONNX sessions"""
        det_path = os.path.join(self.models_dir, "det_10g.onnx")
        rec_path = os.path.join(self.models_dir, "w600k_mbf.onnx")
        
        if not os.path.exists(det_path) or not os.path.exists(rec_path):
            # Try absolute path fallback if relative fails
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            det_path = os.path.join(base_dir, "models", "det_10g.onnx")
            rec_path = os.path.join(base_dir, "models", "w600k_mbf.onnx")

        print(f"Loading models from: {det_path} and {rec_path}")
        
        providers = ['CPUExecutionProvider'] # Fallback to CPU
        
        try:
            self.det_session = ort.InferenceSession(det_path, providers=providers)
            self.rec_session = ort.InferenceSession(rec_path, providers=providers)
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            raise e

    def preprocess_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Resize and pad image for SCRFD (640x640)
        Returns: (input_tensor, scale, padx, pady)
        """
        target_size = self.det_input_size[0]
        h, w, _ = image.shape
        
        # Calculate scale
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad center
        image_tensor = np.full((target_size, target_size, 3), 128, dtype=np.uint8) # Gray background
        padx = (target_size - new_w) // 2
        pady = (target_size - new_h) // 2
        
        image_tensor[pady:pady+new_h, padx:padx+new_w] = resized
        
        # Normalize (x - 127.5) / 128.0
        blob = (image_tensor.astype(np.float32) - 127.5) / 128.0
        
        # HWC -> BCHW
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale, padx, pady

    def parse_detection(self, outputs: List[np.ndarray], scale: float, padx: float, pady: float) -> List[Dict]:
        """Decode SCRFD outputs"""
        # SCRFD outputs: 3 strides (8, 16, 32), each has score, bbox, kps
        # Order in outputs usually: [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
        # But ONNX Runtime returns meaningful names? No, usually list.
        # We assume specific order or rely on shape. 
        # Checking implementation of TS: keys sorted roughly matches.
        
        # Mapping based on typical SCRFD export
        scores_list = [outputs[0], outputs[1], outputs[2]]
        bboxes_list = [outputs[3], outputs[4], outputs[5]]
        kps_list = [outputs[6], outputs[7], outputs[8]]
        
        strides = [8, 16, 32]
        faces = []
        
        for i, stride in enumerate(strides):
            scores = scores_list[i] # [N, 1] or [N, 2]? TS says 2 anchors?
            bboxes = bboxes_list[i]
            kpss = kps_list[i]
            
            # scores shape: [1, H*W*2, 1] usually or similar
            # Flatten
            scores = scores.flatten()
            
            # Filter by threshold
            indices = np.where(scores > self.det_threshold)[0]
            
            feat_w = self.det_input_size[0] // stride
            feat_h = self.det_input_size[0] // stride
            
            for idx in indices:
                score = scores[idx]
                
                # Grid coords
                anchor_idx = idx % 2 # num_anchors = 2
                grid_idx = idx // 2
                grid_y = grid_idx // feat_w
                grid_x = grid_idx % feat_w
                
                # Decode bbox
                # bbox values are distances to l, t, r, b * stride
                b_idx = idx * 4
                dx = grid_x * stride
                dy = grid_y * stride
                
                # Flattened bbox array access
                x1 = dx - bboxes.flatten()[b_idx] * stride
                y1 = dy - bboxes.flatten()[b_idx+1] * stride
                x2 = dx + bboxes.flatten()[b_idx+2] * stride
                y2 = dy + bboxes.flatten()[b_idx+3] * stride
                
                # Map back to original image
                real_x1 = (x1 - padx) / scale
                real_y1 = (y1 - pady) / scale
                real_x2 = (x2 - padx) / scale
                real_y2 = (y2 - pady) / scale
                
                # Landmarks (5 points * 2 coords)
                k_idx = idx * 10
                landmarks = []
                for k in range(5):
                    kx = dx + kpss.flatten()[k_idx + k*2] * stride
                    ky = dy + kpss.flatten()[k_idx + k*2 + 1] * stride
                    real_kx = (kx - padx) / scale
                    real_ky = (ky - pady) / scale
                    landmarks.append([real_kx, real_ky])
                
                faces.append({
                    "bbox": [real_x1, real_y1, real_x2, real_y2],
                    "score": float(score),
                    "landmarks": landmarks,
                    "quality": min(100.0, ((real_x2 - real_x1) * (real_y2 - real_y1)) / 800.0 * 100.0) # Approx quality
                })
                
        return self.nms(faces)

    def nms(self, faces: List[Dict]) -> List[Dict]:
        """Non-Maximum Suppression"""
        if not faces:
            return []
            
        faces.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        
        while faces:
            best = faces.pop(0)
            keep.append(best)
            
            faces = [f for f in faces if self.iou(best['bbox'], f['bbox']) < self.nms_threshold]
            
        return keep

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def align_face(self, image: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
        """Align face using Similarity Transform"""
        src = np.array(landmarks, dtype=np.float32)
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        # Estimate affine transform (using cv2.estimateAffinePartial2D is robust)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        
        aligned = cv2.warpAffine(image, M, self.rec_input_size)
        return aligned

    def get_embedding(self, aligned_face: np.ndarray) -> List[float]:
        """MobileFaceNet Inference"""
        # Preprocess
        blob = (aligned_face.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1)) # HWC -> BCHW
        blob = np.expand_dims(blob, axis=0)
        
        # Inference
        inputs = {self.rec_session.get_inputs()[0].name: blob}
        embedding = self.rec_session.run(None, inputs)[0]
        
        # Normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding.tolist()

    def process_image(self, image_bytes: bytes) -> List[Dict]:
        """Main pipeline"""
        if not self.det_session:
            self.load_models()
            
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # ONNX models expect RGB
        
        # Detection
        input_tensor, scale, padx, pady = self.preprocess_detection(image)
        det_inputs = {self.det_session.get_inputs()[0].name: input_tensor}
        det_outputs = self.det_session.run(None, det_inputs)
        
        faces = self.parse_detection(det_outputs, scale, padx, pady)
        
        results = []
        for face in faces:
            # 40% Padding logic (matches frontend)
            x1, y1, x2, y2 = face['bbox']
            w, h = x2 - x1, y2 - y1
            padW, padH = w * 0.45, h * 0.45
            
            padded_bbox = [
                max(0, x1 - padW),
                max(0, y1 - padH),
                min(image.shape[1], x2 + padW),
                min(image.shape[0], y2 + padH)
            ]
            
            # Align and Recognize
            try:
                aligned = self.align_face(image, face['landmarks'])
                vector = self.get_embedding(aligned)
                
                # Thumbnail
                # We can skip thumbnail generation for backend or implement it if needed by frontend
                # Frontend expects it for display.
                # Let's generate a small base64 thumbnail.
                thumb_io = cv2.imencode('.jpg', cv2.cvtColor(image[int(padded_bbox[1]):int(padded_bbox[3]), int(padded_bbox[0]):int(padded_bbox[2])], cv2.COLOR_RGB2BGR))[1]
                import base64
                thumbnail = "data:image/jpeg;base64," + base64.b64encode(thumb_io).decode('utf-8')

                results.append({
                    "vector": vector,
                    "bbox": padded_bbox,
                    "score": face['score'],
                    "quality": face['quality'],
                    "thumbnail": thumbnail
                })
            except Exception as e:
                print(f"Failed to process face: {e}")
                
        return results

# Singleton
face_processor_service = FaceProcessor()
