"use client";
import { useState } from 'react';
import { UserPlus, ClipboardCheck } from 'lucide-react';
import AddStudentView from '@/components/AddStudentView';
import AttendanceView from '@/components/AttendanceView';

export default function Home() {
  const [activeView, setActiveView] = useState<'home' | 'add_student' | 'mark_attendance'>('home');

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-950 p-4 md:p-8">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-violet-500/10 rounded-full blur-[100px]"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header */}
        <header className={`mb-12 text-center transition-all duration-500 ${activeView !== 'home' ? 'scale-90 opacity-80' : ''}`}>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-violet-600 dark:from-blue-400 dark:to-violet-400">
            Smart Attendance System
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-lg">
            AI-Powered Face Recognition for Educational Institutes
          </p>
        </header>

        {activeView === 'home' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto animate-fade-in">
            <DashboardCard
              title="Add Student"
              icon={<UserPlus size={40} />}
              description="Register new students with high-quality face data for the recognition engine."
              onClick={() => setActiveView('add_student')}
              color="from-blue-500 to-blue-600"
              shadow="shadow-blue-500/30"
            />
            <DashboardCard
              title="Apply Attendance"
              icon={<ClipboardCheck size={40} />}
              description="Upload classroom photos to automatically mark attendance using AI."
              onClick={() => setActiveView('mark_attendance')}
              color="from-violet-500 to-violet-600"
              shadow="shadow-violet-500/30"
            />
          </div>
        )}

        <div className="transition-all duration-300">
          {activeView === 'add_student' && <AddStudentView onBack={() => setActiveView('home')} />}
          {activeView === 'mark_attendance' && <AttendanceView onBack={() => setActiveView('home')} />}
        </div>
      </div>
    </main>
  );
}

function DashboardCard({ title, icon, description, onClick, color, shadow }: any) {
  return (
    <button
      onClick={onClick}
      className="group relative overflow-hidden glass-card p-8 text-left transition-all hover:translate-y-[-5px] hover:shadow-2xl"
    >
      <div className={`absolute top-0 right-0 p-32 blur-3xl opacity-10 group-hover:opacity-20 transition-opacity bg-gradient-to-br ${color} rounded-full -mr-10 -mt-10`}></div>

      <div className="relative z-10 flex flex-col h-full">
        <div className={`p-4 rounded-2xl w-fit mb-6 bg-gradient-to-br ${color} text-white shadow-lg ${shadow}`}>
          {icon}
        </div>
        <h2 className="text-2xl font-bold mb-3 text-slate-800 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
          {title}
        </h2>
        <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
          {description}
        </p>
      </div>
    </button>
  )
}
