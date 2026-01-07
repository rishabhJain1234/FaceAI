# System Architecture & Documentation (Client-Side Inference Version)

## 🎯 Overview
This project is an AI-powered Student Attendance Management System that uses **Client-Side Facial Recognition** for privacy, speed, and cost-effectiveness. 

By shifting the heavy AI processing (Detection and Recognition) from the server to the faculty's local device browser, we achieve:
- ⚡ **Near-instant analysis** (no high-res image uploads to server)
- 🔒 **Privacy-by-design** (images never leave the local device)
- 💰 **Infinite Scalability** (server only does lightweight embedding matching)
- 🚀 **Free Hosting** (backend is lightweight enough for free cloud tiers)

---

## 🏗️ Technology Stack

### Frontend (AI & User Interface)
- **Framework**: [Next.js](https://nextjs.org/) (React)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **AI Runtime**: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (ORT.js)
- **Hardware Acceleration**: WebGPU, WebGL fallbacks
- **Purpose**: Captures images, processes them locally using AI models, and extracts face embeddings.

### Backend (Server-Side Matching)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Server**: Uvicorn (ASGI)
- **Database**: MongoDB (via `pymongo`)
- **Purpose**: Stores student metadata and face vectors (512-d embeddings), and performs lightweight cosine similarity matching.

---

## 🧠 AI & Computer Vision Pipeline

The system uses a **high-precision hybrid client-side pipeline** running in the browser:
- **Detection**: SCRFD (RetinaFace-based) optimized for ONNX Runtime (`det_10g.onnx`).
- **Recognition**: MobileFaceNet (ArcFace MobileNet-variant) optimized for ONNX Runtime (`w600k_mbf.onnx`).

### ⚙️ Runtime Process
1.  **High-Res Loading**: Internal images are loaded at up to **4096px (4K)** to preserve distant faces.
2.  **Detection Workspace**: AI scans at a fixed **1920x1920** resolution (multiple of 32 for optimal kernel performance).
3.  **Preprocessing & Alignment**: 
    *   **Padding Correction**: Automatic letterbox subtraction to ensure perfect coordinate mapping.
    *   **Image Sharpening**: Laplacian sharpening filter applied to face crops to boost recognition on blurry photos.
4.  **Hardware Acceleration**: Uses **WASM Proxy Workers** (multithreading) for smooth UI while running heavy AI inference.
5.  **Quality Gating**:
    *   **Face Quality Score**: Each detection is assigned a score (0-100) based on its effective pixel resolution.
    *   **Visual Debug**: 40% padded thumbnails are generated for facial verification.

### 🔧 Component Details

#### 1. Face Detection (SCRFD)
*   **Role**: Rapid face localization and 5-point landmark detection.
*   **Format**: ONNX (Runtime-optimized).
*   **Resolution**: 1920x1920 (High-accuracy mode).
*   **Capabilities**: Robust detection of 50+ faces in a single classroom group photo.

#### 2. Face Recognition (MobileFaceNet)
*   **Role**: High-fidelity feature extraction.
*   **Input**: Aligned & **Sharpened** 112x112 headshots.
*   **Output**: 512-dimensional normalized face embeddings.

#### 3. Face Alignment & Enhancement
*   **Alignment**: Custom similarity transform mapping 5 landmarks to a standard template.
*   **Enhancement**: Real-time sharpening filter to recover facial details (eyes, nose) in distant detections.
*   **Padding**: 40% visual padding added to thumbnails for administrator review.

---

## ⚙️ Workflows

### 1. "Add Student" Pipeline (Local First)
1.  **Input**: Faculty captures/uploads a clear student photo in the browser.
2.  **Local Processing**:
    *   **Detection**: Browser detects exactly one face.
    *   **Preprocessing**: Face is aligned and cropped to 112×112.
    *   **Recognition**: Browser generates a **512-dimensional embedding** vector.
3.  **Communication**: The browser sends only the **Name, ID, and Embedding Vector** (a few KB) to the server.
4.  **Storage**: MongoDB stores the vector. **The image is never uploaded.**

### 2. "Mark Attendance" Pipeline (Lightning Fast)
1.  **Input**: Faculty captures/uploads a classroom photo.
2.  **Local Processing**:
    *   **Detection**: Browser scans the entire photo locally.
    *   **Preprocessing**: All detected faces are aligned and cropped.
    *   **Recognition**: Browser generates a list of embeddings (512-d each) for every person.
3.  **Communication**: The browser sends a list of embeddings to the backend.
4.  **Server Matching**:
    *   Backend performs **Cosine Similarity** between input vectors and the database.
    *   **Threshold**: Similarity ≥ **0.45** → Student marked "Present".
5.  **Output**: Response returned with Present/Absent status and any unknown detections.

---

## 📂 Project Structure

```
/
├── backend/
│   ├── main.py                # Lightweight FastAPI entry
│   ├── database.py            # Simple MongoDB connection
│   ├── services/
│   │   └── face_service_client.py # FAST vector matching (NumPy)
│   └── routers/               # API Endpoints
│       ├── student.py
│       └── attendance.py
├── frontend/
│   ├── app/                   # Next.js Pages
│   ├── components/            # UI Components
│   ├── lib/
│   │   └── faceProcessor.ts   # 🧠 The AI Heart (ONNX logic)
│   └── public/
│       └── models/            # 🔴 AI Models (det_10g, w600k_mbf)
└── docs/                      # 📄 Documentation folder
```

---

## 🚀 Efficiency & Privacy Comparison

| Metric | Server-Side (Old) | Client-Side (Current) |
|--------|------------------|----------------------|
| **Image Privacy** | ⚠️ Uploaded to server | ✅ Stays on device |
| **Data Usage** | 5-10MB (Large images) | <10KB (Vectors only) |
| **Server Cost** | High ($25+/mo) | $0 (Free Tier) |
| **Response Time** | 3-5s (Upload + Heavy AI) | 0.5-2s (Local AI) |
| **Scalability** | Limited by CPU/GPU | Virtually Infinite |
