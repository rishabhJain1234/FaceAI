"use client";
import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, Users, UserCheck, UserX, AlertCircle, ArrowLeft } from 'lucide-react';
import { faceProcessor, FaceEmbedding } from '@/lib/faceProcessor';

export default function AttendanceView({ onBack }: { onBack: () => void }) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any | null>(null);
    const [error, setError] = useState('');
    const [progress, setProgress] = useState('');

    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const f = e.target.files[0];
            setFile(f);
            setPreview(URL.createObjectURL(f));
            setResult(null);
            setError('');
            setProgress('');
        }
    };

    const handleApply = async () => {
        if (!file) return;

        setLoading(true);
        setError('');
        setProgress('Initializing AI models...');

        try {
            // Step 1: Initialize face processor
            await faceProcessor.initialize();

            // Step 2: Load and preprocess image
            setProgress('Analyzing classroom photo...');
            const imageData = await (faceProcessor.constructor as any).loadImageData(file);

            // Step 3: Process image (detect + extract all embeddings)
            setProgress('Detecting faces...');
            const embeddings: FaceEmbedding[] = await faceProcessor.processAttendanceImage(
                imageData,
                (current, total) => {
                    setProgress(`Processing face ${current} of ${total}...`);
                }
            );

            if (embeddings.length === 0) {
                throw new Error("No faces detected in the image. Please use a clearer photo of the classroom.");
            }

            setProgress(`Matching ${embeddings.length} students with database...`);

            // Step 4: Send embeddings to server for matching
            const payload = {
                embeddings: embeddings.map(e => ({
                    vector: e.vector,
                    bbox: e.bbox,
                    score: e.score,
                    quality: e.quality,
                    thumbnail: e.thumbnail
                }))
            };
            console.log('📤 Sending attendance payload:', payload);

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await axios.post(`${apiUrl}/attendance/mark`, payload);

            console.log(res.data);
            setResult(res.data);
            setProgress('');
        } catch (err: any) {
            console.error(err);
            const detail = err.response?.data?.detail;
            const msg = typeof detail === 'string'
                ? detail
                : (Array.isArray(detail) ? detail.map((d: any) => d.msg).join(', ') : JSON.stringify(detail));

            setError(msg || err.message || 'Failed to mark attendance');
            setProgress('');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto animate-fade-in relative">
            <div className="mb-6">
                <button onClick={onBack} className="flex items-center gap-2 text-slate-500 hover:text-slate-800 dark:hover:text-white transition-colors">
                    <ArrowLeft size={20} /> Back to Dashboard
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Panel: Upload */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="glass-card p-6">
                        <h2 className="text-xl font-bold mb-4 text-slate-800 dark:text-white">Upload Classroom Image</h2>
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all ${preview ? 'border-violet-500 bg-violet-50/10' : 'border-slate-300 hover:border-violet-500 hover:bg-violet-50/10'} min-h-[300px]`}
                        >
                            {preview ? (
                                <img src={preview} alt="Classroom" className="w-full h-full object-contain rounded-lg" />
                            ) : (
                                <>
                                    <Upload className="text-slate-400 mb-4" size={40} />
                                    <p className="text-center text-slate-500">Click to upload image containing students</p>
                                </>
                            )}
                            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                        </div>

                        <button
                            onClick={handleApply}
                            disabled={loading || !file}
                            className="w-full mt-6 py-3 bg-violet-600 hover:bg-violet-700 disabled:opacity-50 text-white rounded-xl font-bold shadow-lg shadow-violet-500/20 transition-all flex items-center justify-center gap-2"
                        >
                            {loading ? <Loader2 className="animate-spin" /> : 'Analyze Attendance'}
                        </button>
                        {progress && <p className="mt-4 text-blue-500 text-sm text-center animate-pulse">{progress}</p>}
                        {error && <p className="mt-4 text-red-500 text-sm text-center">{error}</p>}
                    </div>
                </div>

                {/* Right Panel: Results */}
                <div className="lg:col-span-2 space-y-6">
                    {!result && !loading && (
                        <div className="glass-card p-12 flex flex-col items-center justify-center text-slate-400 h-full min-h-[400px]">
                            <Users size={64} className="mb-4 opacity-50" />
                            <p className="text-lg">Upload an image and click "Analyze" to see results.</p>
                        </div>
                    )}

                    {loading && (
                        <div className="glass-card p-12 flex flex-col items-center justify-center text-slate-400 h-full min-h-[400px]">
                            <Loader2 size={64} className="animate-spin mb-4 text-violet-500" />
                            <p className="text-lg animate-pulse">{progress || 'Processing Face Recognition Pipeline...'}</p>
                            <p className="text-sm mt-2 text-slate-500">Local Detection • WebGPU Acceleration • Privacy First</p>
                        </div>
                    )}

                    {result && (
                        <div className="space-y-6 animate-fade-in">
                            {/* Summary Cards */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <SummaryCard label="Registered" value={result.total_registered} icon={<Users size={20} />} color="bg-blue-500" />
                                <SummaryCard label="Detected" value={result.total_detected} icon={<Users size={20} />} color="bg-amber-500" />
                                <SummaryCard label="Present" value={result.present_students.length} icon={<UserCheck size={20} />} color="bg-green-500" />
                                <SummaryCard label="Unknown" value={result.unknown_count} icon={<AlertCircle size={20} />} color="bg-red-500" />
                            </div>

                            {/* Detailed List */}
                            <div className="glass-card p-6 overflow-hidden">
                                <h3 className="text-lg font-bold mb-4 text-slate-800 dark:text-white">Attendance Report</h3>
                                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                                    <table className="w-full text-left border-collapse">
                                        <thead className="sticky top-0 bg-white/50 backdrop-blur-sm z-10">
                                            <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-500 text-sm">
                                                <th className="p-3">Student Name</th>
                                                <th className="p-3">Status</th>
                                                <th className="p-3">ID Prefix</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                                            {result.present_students.map((s: any) => (
                                                <tr key={s.student_id} className="group hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                                    <td className="p-3 font-medium text-slate-800 dark:text-slate-200">{s.name}</td>
                                                    <td className="p-3">
                                                        <span className="px-3 py-1 rounded-full bg-green-100 text-green-700 text-xs font-bold uppercase dark:bg-green-900/30 dark:text-green-400">Present</span>
                                                    </td>
                                                    <td className="p-3 text-slate-400 text-xs font-mono">{s.student_id.slice(0, 8)}</td>
                                                </tr>
                                            ))}
                                            {result.absent_students.map((s: any) => (
                                                <tr key={s.student_id} className="group hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                                    <td className="p-3 font-medium text-slate-500 dark:text-slate-400">{s.name}</td>
                                                    <td className="p-3">
                                                        <span className="px-3 py-1 rounded-full bg-red-100 text-red-700 text-xs font-bold uppercase dark:bg-red-900/30 dark:text-red-400">Absent</span>
                                                    </td>
                                                    <td className="p-3 text-slate-400 text-xs font-mono">{s.student_id.slice(0, 8)}</td>
                                                </tr>
                                            ))}
                                            {result.present_students.length === 0 && result.absent_students.length === 0 && (
                                                <tr>
                                                    <td colSpan={3} className="p-6 text-center text-slate-500">No registered students found in database.</td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Match Details (Debug) */}
                            {result.matches && result.matches.length > 0 && (
                                <div className="glass-card p-6">
                                    <h3 className="text-lg font-bold mb-4 text-slate-800 dark:text-white flex items-center gap-2">
                                        <AlertCircle size={20} className="text-amber-500" />
                                        Inference Details
                                    </h3>
                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                        {result.matches.map((match: any, i: number) => (
                                            <div key={i} className={`p-3 rounded-xl border ${match.name === 'Unknown' ? 'border-red-200 bg-red-50 dark:bg-red-900/10' : 'border-green-200 bg-green-50 dark:bg-green-900/10'} flex flex-col gap-2`}>
                                                {match.thumbnail && (
                                                    <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-200 border border-slate-300 dark:border-slate-700">
                                                        <img src={match.thumbnail} alt="Face" className="w-full h-full object-cover" />
                                                    </div>
                                                )}
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-bold truncate text-slate-700 dark:text-slate-300" title={match.name}>{match.name}</span>
                                                    <div className="flex items-center justify-between mt-1">
                                                        <span className="text-[10px] text-slate-500 font-mono">Sim: {match.score}</span>
                                                        <span className={`text-[10px] font-mono ${match.quality < 30 ? 'text-red-500 font-bold' : 'text-slate-400'}`}>
                                                            Qual: {match.quality}
                                                        </span>
                                                    </div>
                                                    {match.quality < 30 && (
                                                        <div className="mt-1 px-2 py-0.5 bg-red-100 text-red-700 text-[8px] font-bold rounded uppercase text-center">
                                                            Low Resolution
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

function SummaryCard({ label, value, icon, color }: any) {
    return (
        <div className="glass-card p-4 flex flex-col items-center justify-center text-center">
            <div className={`${color} p-2 rounded-full text-white mb-2 shadow-lg`}>
                {icon}
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-white">{value}</div>
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wide">{label}</div>
        </div>
    )
}
