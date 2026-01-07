# 🚀 Deployment Guide (Client-Side AI Version)

This guide covers deploying the high-efficiency versions of the Attendance System to **Vercel** (Frontend) and **Render** (Backend).

---

## 🏗️ Architecture Overview
- **Frontend (Vercel)**: Runs the 1920px AI Detection & Recognition models locally in the browser.
- **Backend (Render)**: Lightweight FastAPI server that handles only MongoDB storage and 512-d vector matching.
- **Database (MongoDB Atlas)**: Stores student metadata and face signatures.

---

## 🗄️ Step 1: MongoDB Atlas Setup
1. Create a **FREE** Cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. **Network Access**: Add `0.0.0.0/0` (Allow all IPs) so Render can connect.
3. **Database Access**: Create a user (e.g., `admin`) and save the password.
4. **Connection String**: Copy the string: `mongodb+srv://admin:<password>@cluster.mongodb.net/attendance_db`.

---

## 🖥️ Step 2: Backend (Render.com)
The backend is now extremely lightweight (no TensorFlow/ONNX needed).

1. **GitHub**: Push your code to a GitHub repository.
2. **New Web Service**: Connect your repo to Render.
3. **Root Directory**: `backend`
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Environment Variables**:
   - `MONGODB_URI`: Your Atlas connection string.
   - `DB_NAME`: `attendance_db`
   - `PYTHON_VERSION`: `3.13.0`

---

## 🌐 Step 3: Frontend (Vercel.com)
1. **New Project**: Import your repo to Vercel.
2. **Root Directory**: `frontend`
3. **Framework**: Next.js (Auto-detected).
4. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL`: Use your Render URL (e.g., `https://attendance-api.onrender.com`).
5. **Deploy**: Click Deploy. 

---

## 🛠️ Performance Checklist
- [ ] **Hardware Acceleration**: The system automatically uses **WebGPU** or **WASM Proxy Workers** for smooth performance.
- [ ] **CORS**: The backend is pre-configured to allow all origins, ensuring Vercel can talk to Render.
- [ ] **AI Models**: The 30MB models are stored in `frontend/public/models/` and will be cached by the user's browser automatically.

---

## 💡 Troubleshooting
- **Stuck at "Registering..."**: Ensure your `NEXT_PUBLIC_API_URL` environment variable doesn't have a trailing slash.
- **Status 422**: Check if you have cleared the database using `cleanup_db.py` before deploying, as old signatures are incompatible.
- **Unknown Faces**: Ensure the faculty is using high-res photos (the system now supports up to 4K internally).
