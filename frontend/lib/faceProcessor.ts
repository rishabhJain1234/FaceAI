/**
 * Face Processing Service - Client-Side Inference
 * 
 * Handles face detection and recognition using ONNX Runtime Web
 * Models: RetinaFace-Mobile (detection) + MobileFaceNet (recognition)
 * Runtime: WebGPU → WebGL → WASM (automatic fallback)
 */

import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
// Proxy mode runs ONNX in a web worker, which is essential to keep the 
// UI responsive during heavy inference sessions.
ort.env.wasm.proxy = true;
ort.env.logLevel = 'warning';

interface FaceDetection {
    bbox: number[]; // [x1, y1, x2, y2]
    landmarks: number[][]; // [[x,y], [x,y], ...] 5 points
    score: number;
    quality: number; // Quality score (0-100) based on resolution
}

interface FaceEmbedding {
    vector: number[];
    bbox: number[];
    score: number;
    quality?: number; // Face resolution quality
    thumbnail?: string; // Base64 thumbnail for visual debug
}

class FaceProcessor {
    private detectionSession: ort.InferenceSession | null = null;
    private recognitionSession: ort.InferenceSession | null = null;
    private initialized: boolean = false;
    private initPromise: Promise<void> | null = null;

    /**
     * Check if a model file exists and is accessible
     */
    private async verifyModel(path: string): Promise<boolean> {
        try {
            const response = await fetch(path, { method: 'HEAD' });
            if (!response.ok) {
                console.error(`❌ Model file not found: ${path} (Status: ${response.status})`);
                return false;
            }
            return true;
        } catch (e) {
            console.error(`❌ Error verifying model path: ${path}`, e);
            return false;
        }
    }

    /**
     * Initialize ONNX models (lazy loading)
     */
    async initialize(): Promise<void> {
        if (this.initialized) return;
        if (this.initPromise) return this.initPromise;

        this.initPromise = (async () => {
            try {
                const detPath = '/models/det_10g.onnx';
                const recPath = '/models/w600k_mbf.onnx'; // Lightweight MobileNet variant

                console.log('🔄 Verifying model paths...');
                const detOk = await this.verifyModel(detPath);
                const recOk = await this.verifyModel(recPath);

                if (!detOk || !recOk) {
                    throw new Error('One or more model files are missing from public/models/');
                }

                // Helper to create session with stable WASM provider
                const createSession = async (path: string, name: string, provider: string[] = ['wasm']) => {
                    console.log(`🔄 Loading ${name} with ${provider.join(',')}...`);
                    try {
                        return await ort.InferenceSession.create(path, {
                            executionProviders: provider,
                            graphOptimizationLevel: 'all',
                        });
                    } catch (e: any) {
                        console.error(`❌ Failed to load ${name}:`, e);
                        throw e;
                    }
                };

                // SCRFD (Detection) MUST use WASM because WebGPU currently 
                // doesn't support the 'AveragePool with ceil' operation it uses.
                this.detectionSession = await createSession(detPath, 'Detection (SCRFD)', ['wasm']);

                // MobileFaceNet is small enough that WASM is also very fast and more stable
                this.recognitionSession = await createSession(recPath, 'Recognition (MobileFaceNet)', ['wasm']);

                this.initialized = true;
                console.log('✅ Models loaded successfully using WASM');
                console.log(`   Detection Inputs: ${this.detectionSession.inputNames}`);
                console.log(`   Recognition Inputs: ${this.recognitionSession.inputNames}`);
            } catch (error: any) {
                console.error('❌ Failed to initialize models:', error);
                throw new Error(`Model Initialization Failed: ${error.message || error}`);
            }
        })();

        return this.initPromise;
    }

    /**
     * Detect faces in an image
     */
    async detectFaces(imageData: ImageData, threshold: number = 0.7): Promise<FaceDetection[]> {
        if (!this.initialized) {
            await this.initialize();
        }

        try {
            // Preprocess image for detection
            const { tensor, scale, offsetX, offsetY } = this.preprocessForDetection(imageData);

            // Run detection
            const inputName = this.detectionSession!.inputNames[0];
            const feeds = { [inputName]: tensor };
            const results = await this.detectionSession!.run(feeds);

            // Parse results
            const faces = this.parseDetectionResults(results, threshold, scale, offsetX, offsetY);

            console.log(`✅ Detected ${faces.length} face(s)`);
            return faces;
        } catch (error) {
            console.error('❌ Face detection failed:', error);
            throw error;
        }
    }

    /**
     * Extract embedding for a single face
     */
    async extractEmbedding(
        imageData: ImageData,
        bbox: number[],
        landmarks: number[][]
    ): Promise<number[]> {
        if (!this.initialized) {
            await this.initialize();
        }

        try {
            // Align face using landmarks
            const alignedFace = this.alignFace(imageData, landmarks);

            // Preprocess for recognition
            const inputTensor = this.preprocessForRecognition(alignedFace);

            // Run recognition
            const inputName = this.recognitionSession!.inputNames[0];
            const feeds = { [inputName]: inputTensor };
            const results = await this.recognitionSession!.run(feeds);

            // Extract and normalize embedding (get first output available)
            const firstOutput = Object.values(results)[0];
            const embedding = Array.from(firstOutput.data as Float32Array);
            console.log(`📡 Generated embedding with length: ${embedding.length}`);
            return this.normalizeEmbedding(embedding);
        } catch (error) {
            console.error('❌ Embedding extraction failed:', error);
            throw error;
        }
    }

    /**
     * Process image for attendance (detect + extract all embeddings)
     */
    async processAttendanceImage(
        imageData: ImageData,
        onProgress?: (current: number, total: number) => void
    ): Promise<FaceEmbedding[]> {
        // Lower threshold (0.5) to catch smaller faces in the back
        const faces = await this.detectFaces(imageData, 0.5);

        if (faces.length === 0) {
            return [];
        }

        // Extract embeddings for all faces
        const embeddings: FaceEmbedding[] = [];
        const total = faces.length;

        for (let i = 0; i < total; i++) {
            const face = faces[i];
            if (onProgress) onProgress(i + 1, total);

            // Apply 40% Padding to the bounding box
            const [x1, y1, x2, y2] = face.bbox;
            const w = x2 - x1;
            const h = y2 - y1;
            const padW = w * 0.45; // slightly more for 40% effective padding
            const padH = h * 0.45;

            const paddedBbox = [
                Math.max(0, x1 - padW),
                Math.max(0, y1 - padH),
                Math.min(imageData.width, x2 + padW),
                Math.min(imageData.height, y2 + padH)
            ];

            // Yeild to main thread to keep UI alive (especially if proxy is somehow not used)
            await new Promise(resolve => setTimeout(resolve, 0));

            try {
                // Recognition still uses landmarks for precise alignment
                const vector = await this.extractEmbedding(imageData, face.bbox, face.landmarks);

                // RESTORE VISUAL DEBUG: Capture thumbnail with the new PADDED bbox
                let thumbnail = '';
                try {
                    thumbnail = this.captureFaceThumbnail(imageData, paddedBbox);
                } catch (e) {
                    console.warn('⚠️ Thumbnail capture failed', e);
                }

                embeddings.push({
                    vector,
                    bbox: paddedBbox,
                    score: face.score,
                    quality: face.quality,
                    thumbnail
                });
            } catch (error) {
                console.warn(`⚠️ Failed to extract embedding for face ${i + 1}:`, error);
            }
        }

        console.log(`✅ Extracted ${embeddings.length} embedding(s)`);
        return embeddings;
    }

    private preprocessForDetection(imageData: ImageData): { tensor: ort.Tensor, scale: number, offsetX: number, offsetY: number } {
        const targetSize = 1920; // Correct multiple of 32 for high-res detection
        const { width, height } = imageData;

        // Calculate scale to fit within targetSize while maintaining aspect ratio
        const scale = Math.min(targetSize / width, targetSize / height);
        const newWidth = Math.round(width * scale);
        const newHeight = Math.round(height * scale);
        const offsetX = (targetSize - newWidth) / 2;
        const offsetY = (targetSize - newHeight) / 2;

        // Create canvas for resizing
        const canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext('2d')!;

        // Fill with gray background
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Create temporary canvas with original image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.putImageData(imageData, 0, 0);

        // Draw resized image centered
        ctx.drawImage(tempCanvas, 0, 0, width, height, offsetX, offsetY, newWidth, newHeight);

        // Get image data and convert to tensor
        const resizedImageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const data = resizedImageData.data;

        // Convert to CHW format and normalize (InsightFace standard: / 128.0)
        const float32Data = new Float32Array(3 * targetSize * targetSize);
        for (let i = 0; i < targetSize * targetSize; i++) {
            float32Data[i] = (data[i * 4] - 127.5) / 128.0; // R
            float32Data[targetSize * targetSize + i] = (data[i * 4 + 1] - 127.5) / 128.0; // G
            float32Data[2 * targetSize * targetSize + i] = (data[i * 4 + 2] - 127.5) / 128.0; // B
        }

        return {
            tensor: new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]),
            scale,
            offsetX,
            offsetY
        };
    }

    /**
     * Preprocess aligned face for recognition (112x112)
     */
    private preprocessForRecognition(imageData: ImageData): ort.Tensor {
        const targetSize = 112;

        // Create canvas for resizing
        const canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext('2d')!;

        // Create temporary canvas with face image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imageData.width;
        tempCanvas.height = imageData.height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.putImageData(imageData, 0, 0);

        // Draw resized
        ctx.drawImage(tempCanvas, 0, 0, targetSize, targetSize);

        // Get image data and convert to tensor
        const resizedImageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const data = resizedImageData.data;

        // Convert to CHW format and normalize (InsightFace standard: / 128.0)
        const float32Data = new Float32Array(3 * targetSize * targetSize);
        for (let i = 0; i < targetSize * targetSize; i++) {
            float32Data[i] = (data[i * 4] - 127.5) / 128.0; // R
            float32Data[targetSize * targetSize + i] = (data[i * 4 + 1] - 127.5) / 128.0; // G
            float32Data[2 * targetSize * targetSize + i] = (data[i * 4 + 2] - 127.5) / 128.0; // B
        }

        return new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
    }

    /**
     * Parse SCRFD (RetinaFace) detection model outputs
     * SCRFD det_10g has 9 outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
     */
    private parseDetectionResults(
        results: ort.InferenceSession.OnnxValueMapType,
        threshold: number,
        scale: number,
        offsetX: number,
        offsetY: number
    ): FaceDetection[] {
        const targetSize = 1920; // Must match preprocessForDetection
        const strides = [8, 16, 32];
        const numAnchors = 2; // SCRFD typically uses 2 anchors per scale

        // Map output keys to strides (order: 0-2=scores, 3-5=bboxes, 6-8=kps)
        const keys = Object.keys(results).sort((a, b) => parseInt(a) - parseInt(b));

        let candidates: FaceDetection[] = [];

        for (let i = 0; i < strides.length; i++) {
            const stride = strides[i];
            const scoreData = results[keys[i * 3]].data as Float32Array;
            const bboxData = results[keys[i * 3 + 1]].data as Float32Array;
            const kpsData = results[keys[i * 3 + 2]].data as Float32Array;

            const featSize = (targetSize / stride) * (targetSize / stride);
            const featW = targetSize / stride;

            for (let idx = 0; idx < featSize; idx++) {
                for (let a = 0; a < numAnchors; a++) {
                    const scoreIdx = idx * numAnchors + a;
                    const score = scoreData[scoreIdx];

                    if (score > threshold) {
                        const row = Math.floor(idx / featW);
                        const col = idx % featW;

                        const dx = col * stride;
                        const dy = row * stride;

                        // Decode bbox (distance to 4 edges * stride) in 640 space
                        const bIdx = scoreIdx * 4;
                        const nx1 = dx - bboxData[bIdx] * stride;
                        const ny1 = dy - bboxData[bIdx + 1] * stride;
                        const nx2 = dx + bboxData[bIdx + 2] * stride;
                        const ny2 = dy + bboxData[bIdx + 3] * stride;

                        // Map back to original image coordinates: (nx - offset) / scale
                        // CLIP to ensures we stay within the real image area
                        const x1 = Math.max(0, (nx1 - offsetX) / scale);
                        const y1 = Math.max(0, (ny1 - offsetY) / scale);
                        const x2 = Math.max(0, (nx2 - offsetX) / scale);
                        const y2 = Math.max(0, (ny2 - offsetY) / scale);

                        // Calculate quality score based on face area in 1920 space
                        const faceW = nx2 - nx1;
                        const faceH = ny2 - ny1;
                        const quality = Math.min(100, Math.round((faceW * faceH) / 800)); // ~28x28 is minimum acceptable

                        // Decode landmarks
                        const kIdx = scoreIdx * 10;
                        const landmarks: number[][] = [];
                        for (let k = 0; k < 5; k++) {
                            const lnx = dx + kpsData[kIdx + k * 2] * stride;
                            const lny = dy + kpsData[kIdx + k * 2 + 1] * stride;
                            landmarks.push([
                                Math.max(0, (lnx - offsetX) / scale),
                                Math.max(0, (lny - offsetY) / scale)
                            ]);
                        }

                        candidates.push({
                            bbox: [x1, y1, x2, y2],
                            landmarks,
                            score,
                            quality
                        });
                    }
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS) - 0.5 for better crowded handling
        return this.applyNMS(candidates, 0.5);
    }

    /**
     * Standard Non-Maximum Suppression
     */
    private applyNMS(candidates: FaceDetection[], iouThreshold: number): FaceDetection[] {
        const sorted = candidates.sort((a, b) => b.score - a.score);
        const selected: FaceDetection[] = [];
        const active = new Array(sorted.length).fill(true);

        for (let i = 0; i < sorted.length; i++) {
            if (!active[i]) continue;

            selected.push(sorted[i]);

            for (let j = i + 1; j < sorted.length; j++) {
                if (!active[j]) continue;

                if (this.calculateIoU(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
                    active[j] = false;
                }
            }
        }

        return selected;
    }

    private calculateIoU(boxA: number[], boxB: number[]): number {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[2], boxB[2]);
        const yB = Math.min(boxA[3], boxB[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

        return interArea / (areaA + boxBArea - interArea + 1e-6);
    }

    /**
     * Align face using 5-point landmarks
     */
    private alignFace(imageData: ImageData, landmarks: number[][]): ImageData {
        if (landmarks.length !== 5) {
            throw new Error('Expected 5 landmarks for face alignment');
        }

        // Standard 5-point template (normalized to 112x112)
        const template = [
            [38.2946, 51.6963], // Left eye
            [73.5318, 51.5014], // Right eye
            [56.0252, 71.7366], // Nose
            [41.5493, 92.3655], // Left mouth
            [70.7299, 92.2041], // Right mouth
        ];

        // Calculate affine transform
        const transform = this.estimateAffineTransform(landmarks, template);

        // Apply transform
        const outputSize = 112;
        const canvas = document.createElement('canvas');
        canvas.width = outputSize;
        canvas.height = outputSize;
        const ctx = canvas.getContext('2d')!;

        // Create source canvas
        const srcCanvas = document.createElement('canvas');
        srcCanvas.width = imageData.width;
        srcCanvas.height = imageData.height;
        const srcCtx = srcCanvas.getContext('2d')!;
        srcCtx.putImageData(imageData, 0, 0);

        // Apply transform
        ctx.setTransform(
            transform[0][0], transform[1][0],
            transform[0][1], transform[1][1],
            transform[0][2], transform[1][2]
        );
        ctx.drawImage(srcCanvas, 0, 0);

        // RESTORE ACCURACY: Sharpening Filter (Unsharp Mask)
        // Helps recognition model see features in blurry group photos
        this.applySharpen(ctx, outputSize);

        return ctx.getImageData(0, 0, outputSize, outputSize);
    }

    private applySharpen(ctx: CanvasRenderingContext2D, size: number) {
        const imgData = ctx.getImageData(0, 0, size, size);
        const data = imgData.data;
        const copy = new Uint8ClampedArray(data);
        const amount = 0.5; // Sharpening strength

        for (let i = 0; i < data.length; i += 4) {
            if (i < size * 4 || i > data.length - size * 4) continue;
            // Simple Laplacian sharpen
            for (let c = 0; c < 3; c++) {
                const center = copy[i + c];
                const up = copy[i + c - size * 4];
                const down = copy[i + c + size * 4];
                const left = copy[i + c - 4];
                const right = copy[i + c + 4];

                const val = center + (center * 4 - up - down - left - right) * amount;
                data[i + c] = Math.max(0, Math.min(255, val));
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    /**
     * Estimate affine transform (Similarity Transform) from source to destination points
     * used for face alignment.
     */
    private estimateAffineTransform(src: number[][], dst: number[][]): number[][] {
        // Implementation of Similarity Transform (Rotation, Scale, Translation)
        // src: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]] (landmarks)
        // dst: [[x1',y1'], [x2',y2'], [x3',y3'], [x4',y4'], [x5',y5']] (template)

        let numPoints = src.length;
        let srcMeanX = 0, srcMeanY = 0, dstMeanX = 0, dstMeanY = 0;

        for (let i = 0; i < numPoints; i++) {
            srcMeanX += src[i][0];
            srcMeanY += src[i][1];
            dstMeanX += dst[i][0];
            dstMeanY += dst[i][1];
        }

        srcMeanX /= numPoints;
        srcMeanY /= numPoints;
        dstMeanX /= numPoints;
        dstMeanY /= numPoints;

        let srcVar = 0, srcCovX = 0, srcCovY = 0;
        for (let i = 0; i < numPoints; i++) {
            let dx = src[i][0] - srcMeanX;
            let dy = src[i][1] - srcMeanY;
            srcVar += dx * dx + dy * dy;
            srcCovX += dx * (dst[i][0] - dstMeanX) + dy * (dst[i][1] - dstMeanY);
            srcCovY += dx * (dst[i][1] - dstMeanY) - dy * (dst[i][0] - dstMeanX);
        }

        if (srcVar < 1e-6) return [[1, 0, 0], [0, 1, 0]];

        let a = srcCovX / srcVar;
        let b = srcCovY / srcVar;

        let tx = dstMeanX - (a * srcMeanX + b * srcMeanY);
        let ty = dstMeanY - (a * srcMeanY - b * srcMeanX);

        // Return affine matrix [ [a, b, tx], [-b, a, ty] ]
        return [
            [a, b, tx],
            [-b, a, ty]
        ];
    }

    /**
     * Normalize embedding vector (L2 normalization)
     */
    private normalizeEmbedding(embedding: number[]): number[] {
        const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        if (norm === 0) return embedding;
        return embedding.map((val) => val / norm);
    }

    /**
     * Load image from File object
     */
    static async loadImageData(file: File): Promise<ImageData> {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Increase internal resolution (4096) to preserve faces in group photos
                const maxWidth = 4096;
                let width = img.width;
                let height = img.height;

                if (width > maxWidth) {
                    const scale = maxWidth / width;
                    width = maxWidth;
                    height = Math.round(height * scale);
                }

                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d')!;
                ctx.drawImage(img, 0, 0, width, height);

                const imageData = ctx.getImageData(0, 0, width, height);
                resolve(imageData);
            };
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });
    }

    /**
     * Capture a base64 thumbnail of the detected face area
     */
    private captureFaceThumbnail(imageData: ImageData, bbox: number[]): string {
        // Clamp bbox to image boundaries
        const x1 = Math.max(0, Math.min(bbox[0], imageData.width - 1));
        const y1 = Math.max(0, Math.min(bbox[1], imageData.height - 1));
        const x2 = Math.max(x1 + 1, Math.min(bbox[2], imageData.width));
        const y2 = Math.max(y1 + 1, Math.min(bbox[3], imageData.height));

        const width = x2 - x1;
        const height = y2 - y1;

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d')!;

        // Performance: Use a temporary canvas to draw the full image once
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imageData.width;
        tempCanvas.height = imageData.height;
        tempCanvas.getContext('2d')!.putImageData(imageData, 0, 0);

        ctx.drawImage(tempCanvas, x1, y1, width, height, 0, 0, width, height);

        return canvas.toDataURL('image/jpeg', 0.8);
    }
}

// Export singleton instance
export const faceProcessor = new FaceProcessor();
export type { FaceDetection, FaceEmbedding };
