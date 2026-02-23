"use client";
import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, Check, X, ArrowLeft } from 'lucide-react';
// import { faceProcessor } from '@/lib/faceProcessor';
import { v4 as uuidv4 } from 'uuid';

export default function AddStudentView({ onBack }: { onBack: () => void }) {
    const [name, setName] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [message, setMessage] = useState('');
    const [progress, setProgress] = useState('');

    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const f = e.target.files[0];
            setFile(f);
            setPreview(URL.createObjectURL(f));
            setStatus('idle');
            setMessage('');
            setProgress('');
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file || !name) return;

        setLoading(true);
        setStatus('idle');
        setMessage('');
        setProgress('Uploading student data...');

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('name', name);
            formData.append('student_id', uuidv4());

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

            setProgress('Processing on server...');
            await axios.post(`${apiUrl}/students/add`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setStatus('success');
            setMessage(`Student ${name} added successfully!`);
            setName('');
            setFile(null);
            setPreview(null);
            setProgress('');
        } catch (err: any) {
            console.error(err);
            const detail = err.response?.data?.detail;
            const msg = typeof detail === 'string'
                ? detail
                : (Array.isArray(detail) ? detail.map((d: any) => d.msg).join(', ') : JSON.stringify(detail));

            setStatus('error');
            setMessage(msg || err.message || 'Failed to add student');
            setProgress('');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-card max-w-2xl mx-auto p-8 animate-fade-in relative">
            <button onClick={onBack} className="absolute top-6 left-6 p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                <ArrowLeft size={24} className="text-slate-600 dark:text-slate-300" />
            </button>

            <h2 className="text-3xl font-bold text-center mb-8 text-slate-800 dark:text-white">Add New Student</h2>

            <form onSubmit={handleSubmit} className="space-y-8">
                <div>
                    <label className="block text-sm font-semibold mb-2 text-slate-600 dark:text-slate-300">Student Full Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="w-full p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-900/50 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all placeholder:text-slate-400"
                        placeholder="e.g. John Doe"
                        required
                    />
                </div>

                <div className="space-y-2">
                    <label className="block text-sm font-semibold mb-2 text-slate-600 dark:text-slate-300">Face Photo</label>
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${preview ? 'border-green-500 bg-green-50/10' : 'border-slate-300 hover:border-blue-500 hover:bg-blue-50/10'}`}
                    >
                        {preview ? (
                            <div className="relative group">
                                <img src={preview} alt="Preview" className="h-48 w-48 object-cover rounded-full shadow-xl ring-4 ring-white dark:ring-slate-800" />
                                <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-full opacity-0 group-hover:opacity-100 transition-opacity text-white font-medium">Change</div>
                            </div>
                        ) : (
                            <>
                                <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-full mb-4 text-blue-600 dark:text-blue-400">
                                    <Upload size={32} />
                                </div>
                                <p className="text-slate-600 dark:text-slate-300 font-medium">Click to upload photo</p>
                                <p className="text-slate-400 text-sm mt-1">Single clear face image (JPEG/PNG)</p>
                            </>
                        )}
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                    </div>
                </div>

                {progress && (
                    <div className="text-center text-blue-600 dark:text-blue-400 font-medium animate-pulse">
                        {progress}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading || !file || !name}
                    className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold text-lg shadow-lg shadow-blue-500/20 transition-all flex items-center justify-center gap-2 transform active:scale-95"
                >
                    {loading ? (
                        <>
                            <Loader2 className="animate-spin" /> {progress || 'Processing...'}
                        </>
                    ) : 'Register Student'}
                </button>
            </form>

            {status === 'success' && (
                <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 rounded-xl flex items-center gap-3 animate-fade-in">
                    <div className="bg-green-100 dark:bg-green-800 p-1 rounded-full"><Check size={16} /></div>
                    <span className="font-medium">{message}</span>
                </div>
            )}
            {status === 'error' && (
                <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 rounded-xl flex items-center gap-3 animate-fade-in">
                    <div className="bg-red-100 dark:bg-red-800 p-1 rounded-full"><X size={16} /></div>
                    <span className="font-medium">{message}</span>
                </div>
            )}
        </div>
    )
}
