import sys
import os
import cv2
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.face_processor import face_processor_service

def test_inference():
    print("Testing FaceProcessor...")
    try:
        # Explicitly load to check paths
        face_processor_service.load_models()
        print("✅ Models loaded successfully")
        
        # Create dummy image (640x640 RGB)
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img)
        
        print("Running processing on dummy image...")
        results = face_processor_service.process_image(encoded.tobytes())
        print(f"✅ Processing complete. Found {len(results)} faces (expected 0 for random noise).")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
