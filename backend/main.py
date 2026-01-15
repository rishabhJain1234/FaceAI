from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import student, attendance
import uvicorn

app = FastAPI(title="Attendance Management System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://classattendance-phi.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(student.router, prefix="/students", tags=["Students"])
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])

@app.get("/")
def read_root():
    return {"message": "Attendance System API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
