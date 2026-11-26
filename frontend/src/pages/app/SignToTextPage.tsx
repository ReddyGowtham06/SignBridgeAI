import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Camera,
  CameraOff,
  Video,
  Trash2,
  Copy,
  Check,
  AlertCircle,
  Info,
  ShieldCheck,
  Lightbulb,
  Maximize2,
  RefreshCw,
  Sparkles,
  Activity,
  Hand,
  Cpu,
  Layers,
  Clock,
  CheckCircle2,
  HelpCircle,
  Save,
  Loader2,
} from 'lucide-react'
import { historyApi, ApiError } from '../../services/api'
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision'
import { Button } from '../../components/common/Button'
import { Badge } from '../../components/common/Badge'
import {
  slotMediaPipeResult,
  computeVideoFeatureVector,
  type SlottedFrame,
} from '../../utils/featureBuilder'
import {
  loadModel,
  classifySign,
  TemporalDebouncer,
  type LoadedModel,
  type PredictionResult,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
} from '../../utils/signClassifier'

// 21 standard HandLandmarker skeletal connections
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky finger & palm base
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]

const BUFFER_MAX_FRAMES = 60
const INFERENCE_STRIDE_FRAMES = 15

export type LiveInferenceStatus =
  | 'standby'
  | 'waiting_hands'
  | 'collecting_frames'
  | 'motion_stationary'
  | 'confidence_low'
  | 'static_presence_low'
  | 'static_pose_jitter'
  | 'idle_hand_resting'
  | 'static_confidence_ambiguous'
  | 'recognizing'
  | 'sign_recognized'

export interface ConfirmedSignEntry {
  label: string;
  cleanWord: string;
  confidence: number;
  timestamp: string;
}

/**
 * Formats raw dataset classes into clean, human-readable display words.
 * E.g. "1.bird" -> "Bird", "10.youareperfect" -> "You Are Perfect", "26.NAME" -> "Name".
 */
function formatClassLabel(rawLabel: string): string {
  const stripped = rawLabel.replace(/^\d+\./, '')

  const dictionary: Record<string, string> = {
    youareperfect: 'You Are Perfect',
    goodmorning: 'Good Morning',
    heavyrain: 'Heavy Rain',
    iamfine: 'I Am Fine',
    iloveyou: 'I Love You',
    nicetomeetyou: 'Nice To Meet You',
    seeyoulater: 'See You Later',
    takecare: 'Take Care',
    whatisyourname: 'What Is Your Name',
  }

  const lower = stripped.toLowerCase()
  if (dictionary[lower]) {
    return dictionary[lower]
  }

  return stripped
    .replace(/[_-]/g, ' ')
    .trim()
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export const SignToTextPage = () => {
  // Camera & Device states
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isCameraLoading, setIsCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isMirrored, setIsMirrored] = useState(true)

  // MediaPipe Landmarker states
  const [isLandmarkerLoading, setIsLandmarkerLoading] = useState(true)
  const [landmarkerError, setLandmarkerError] = useState<string | null>(null)
  const [isLandmarkerReady, setIsLandmarkerReady] = useState(false)

  // Classifier Model states
  const [isClassifierLoading, setIsClassifierLoading] = useState(true)
  const [classifierError, setClassifierError] = useState<string | null>(null)
  const [isClassifierReady, setIsClassifierReady] = useState(false)

  // Real-time detection telemetry
  const [detectedHandsCount, setDetectedHandsCount] = useState(0)
  const [handLabels, setHandLabels] = useState<string[]>([])
  const [fps, setFps] = useState(0)

  // Live Inference Telemetry states
  const [inferenceStatus, setInferenceStatus] = useState<LiveInferenceStatus>('standby')
  const [bufferCount, setBufferCount] = useState(0)
  const [latestPrediction, setLatestPrediction] = useState<PredictionResult | null>(null)
  const [confirmedSign, setConfirmedSign] = useState<ConfirmedSignEntry | null>(null)
  const [recentHistory, setRecentHistory] = useState<ConfirmedSignEntry[]>([])
  const [showDebugTelemetry, setShowDebugTelemetry] = useState(false)

  // UI output state
  const [copied, setCopied] = useState(false)
  const [transcript, setTranscript] = useState<string>('')
  const [isSavingTranslation, setIsSavingTranslation] = useState(false)
  const [saveFeedback, setSaveFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  // DOM and Instance refs
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const classifierModelRef = useRef<LoadedModel | null>(null)
  const debouncerRef = useRef<TemporalDebouncer>(new TemporalDebouncer({ minConsecutiveFrames: 2 }))
  const animFrameIdRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const lastTimestampRef = useRef<number>(0)
  const isMirroredRef = useRef<boolean>(true)
  const frameCountRef = useRef<number>(0)
  const lastFpsUpdateRef = useRef<number>(0)

  // Temporal rolling frame buffer & stride counter
  const frameBufferRef = useRef<SlottedFrame[]>([])
  const strideCounterRef = useRef<number>(0)
  const lastCommittedLabelRef = useRef<string | null>(null)
  const prevHandsCountRef = useRef<number>(-1)
  const prevBufferCountRef = useRef<number>(-1)
  const prevHandLabelsKeyRef = useRef<string>('')

  // Keep isMirroredRef in sync
  useEffect(() => {
    isMirroredRef.current = isMirrored
  }, [isMirrored])

  // Draw 21 actual landmarks and connections on canvas overlay
  const drawHandLandmarks = (
    canvas: HTMLCanvasElement,
    video: HTMLVideoElement,
    result: HandLandmarkerResult,
    mirrored: boolean
  ) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Synchronize canvas resolution with video stream
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!result.landmarks || result.landmarks.length === 0) {
      return
    }

    const { width, height } = canvas

    result.landmarks.forEach((handLandmarks, handIndex) => {
      // Hand 0: Electric Cyan Theme, Hand 1: Electric Violet Theme
      const isFirstHand = handIndex === 0
      const strokeColor = isFirstHand ? 'rgba(6, 182, 212, 0.9)' : 'rgba(168, 85, 247, 0.9)'
      const glowColor = isFirstHand ? '#00f0ff' : '#c084fc'
      const jointFill = isFirstHand ? '#22d3ee' : '#a855f7'
      const fingertipColor = isFirstHand ? '#ffffff' : '#f5d0fe'

      // Transform normalized coordinates (accounting for horizontal mirror for visual drawing only)
      const getCoord = (idx: number) => {
        const point = handLandmarks[idx]
        if (!point) return { x: 0, y: 0 }
        const x = mirrored ? (1 - point.x) * width : point.x * width
        const y = point.y * height
        return { x, y }
      }

      // Draw bone connection lines with glowing shadow
      ctx.save()
      ctx.lineWidth = 3
      ctx.strokeStyle = strokeColor
      ctx.shadowColor = glowColor
      ctx.shadowBlur = 10
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = getCoord(startIdx)
        const p2 = getCoord(endIdx)

        ctx.beginPath()
        ctx.moveTo(p1.x, p1.y)
        ctx.lineTo(p2.x, p2.y)
        ctx.stroke()
      }
      ctx.restore()

      // Draw 21 joint nodes
      handLandmarks.forEach((_, idx) => {
        const { x, y } = getCoord(idx)
        const isFingertip = [4, 8, 12, 16, 20].includes(idx)
        const isWrist = idx === 0

        ctx.save()
        if (isFingertip || isWrist) {
          // Outer glowing pulse ring
          ctx.beginPath()
          ctx.arc(x, y, 7, 0, 2 * Math.PI)
          ctx.strokeStyle = glowColor
          ctx.lineWidth = 1.5
          ctx.stroke()

          // Inner solid node
          ctx.beginPath()
          ctx.arc(x, y, 4.5, 0, 2 * Math.PI)
          ctx.fillStyle = fingertipColor
          ctx.fill()
          ctx.strokeStyle = '#050814'
          ctx.lineWidth = 1
          ctx.stroke()
        } else {
          // Regular joint node
          ctx.beginPath()
          ctx.arc(x, y, 3.5, 0, 2 * Math.PI)
          ctx.fillStyle = jointFill
          ctx.fill()
          ctx.strokeStyle = '#050814'
          ctx.lineWidth = 1
          ctx.stroke()
        }
        ctx.restore()
      })

      // Draw Handedness Label tag near wrist (Index 0)
      const wrist = getCoord(0)
      const handednessName =
        result.handedness?.[handIndex]?.[0]?.displayName ||
        result.handedness?.[handIndex]?.[0]?.categoryName ||
        `Hand ${handIndex + 1}`

      ctx.save()
      ctx.font = '600 11px system-ui, -apple-system, sans-serif'
      const labelText = `${handednessName} (21 pts)`
      const textWidth = ctx.measureText(labelText).width
      const labelX = Math.max(10, Math.min(width - textWidth - 20, wrist.x - textWidth / 2))
      const labelY = Math.min(height - 15, wrist.y + 26)

      // Background rounded pill
      ctx.fillStyle = isFirstHand ? 'rgba(8, 47, 73, 0.85)' : 'rgba(59, 7, 100, 0.85)'
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(labelX - 8, labelY - 14, textWidth + 16, 20, 6)
      ctx.fill()
      ctx.stroke()

      // Text label
      ctx.fillStyle = '#f8fafc'
      ctx.fillText(labelText, labelX, labelY)
      ctx.restore()
    })
  }

  const [retryCount, setRetryCount] = useState(0)

  // Initialize both MediaPipe HandLandmarker and ISL Classifier on mount or retry
  useEffect(() => {
    let isMounted = true

    async function init() {
      // 1. Initialize MediaPipe HandLandmarker
      try {
        let vision
        try {
          vision = await FilesetResolver.forVisionTasks('/wasm')
        } catch (wasmErr) {
          console.warn('Local WASM fileset failed, trying CDN fallback:', wasmErr)
          vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
          )
        }

        let landmarker: HandLandmarker
        try {
          landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: '/models/hand_landmarker.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          })
        } catch (gpuErr) {
          console.warn('GPU delegate unavailable, falling back to CPU delegate:', gpuErr)
          landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: '/models/hand_landmarker.task',
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          })
        }

        if (isMounted) {
          landmarkerRef.current = landmarker
          setIsLandmarkerReady(true)
          setIsLandmarkerLoading(false)
        }
      } catch (err: unknown) {
        console.error('Failed to initialize MediaPipe Hand Landmarker:', err)
        if (isMounted) {
          setLandmarkerError(
            'Unable to load the MediaPipe Hand Landmarker model. Please verify that public/models/hand_landmarker.task is accessible.'
          )
          setIsLandmarkerLoading(false)
        }
      }

      // 2. Initialize ISL Classifier Model
      try {
        const classifier = await loadModel('/models/sign_classifier.json')
        if (isMounted) {
          classifierModelRef.current = classifier
          setIsClassifierReady(true)
          setIsClassifierLoading(false)
        }
      } catch (cErr: unknown) {
        console.error('Failed to load ISL classifier model:', cErr)
        if (isMounted) {
          setClassifierError(
            'Failed to load the trained ISL classifier model. Please verify that public/models/sign_classifier.json is accessible.'
          )
          setIsClassifierLoading(false)
        }
      }
    }

    init()

    return () => {
      isMounted = false
      if (landmarkerRef.current) {
        landmarkerRef.current.close()
        landmarkerRef.current = null
      }
    }
  }, [retryCount])

  // Stop camera tracks, cancel animation frame, and reset inference state cleanly
  const stopCamera = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current)
      animFrameIdRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }

    // Reset camera state
    setIsCameraActive(false)
    setIsCameraLoading(false)
    setDetectedHandsCount(0)
    setHandLabels([])
    setFps(0)

    // Reset temporal inference buffers and debouncer
    frameBufferRef.current = []
    strideCounterRef.current = 0
    debouncerRef.current.reset()
    lastCommittedLabelRef.current = null
    prevHandsCountRef.current = -1
    prevBufferCountRef.current = -1
    prevHandLabelsKeyRef.current = ''
    setBufferCount(0)
    setInferenceStatus('standby')
    setLatestPrediction(null)
  }, [])

  // Start webcam stream and initiate the real-time detection & inference loop
  const startCamera = async () => {
    setCameraError(null)
    setIsCameraLoading(true)

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError(
        'Your browser does not support camera access via the MediaDevices API. Please use a modern browser.'
      )
      setIsCameraLoading(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      setIsCameraActive(true)
      setInferenceStatus('collecting_frames')

      // Self-contained detection and live inference loop
      const runDetection = () => {
        const video = videoRef.current
        const canvas = canvasRef.current
        const landmarker = landmarkerRef.current
        const classifier = classifierModelRef.current

        if (video && canvas && landmarker && video.readyState >= 2 && !video.paused) {
          if (video.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = video.currentTime
            const now = performance.now()

            // Enforce monotonically increasing integer timestamps for MediaPipe & velocity deltas
            const timestamp = now > lastTimestampRef.current ? now : lastTimestampRef.current + 1
            lastTimestampRef.current = timestamp

            try {
              const result = landmarker.detectForVideo(video, timestamp)

              // 1. Draw skeletal overlay on canvas
              drawHandLandmarks(canvas, video, result, isMirroredRef.current)

              // 2. Update detection telemetry (only trigger React state when values change)
              const count = result.landmarks ? result.landmarks.length : 0
              if (prevHandsCountRef.current !== count) {
                prevHandsCountRef.current = count
                setDetectedHandsCount(count)
              }

              const labels =
                result.handedness && result.handedness.length > 0
                  ? result.handedness.map(
                      (h) => h[0]?.displayName || h[0]?.categoryName || 'Hand'
                    )
                  : []
              const labelsKey = labels.join('|')
              if (prevHandLabelsKeyRef.current !== labelsKey) {
                prevHandLabelsKeyRef.current = labelsKey
                setHandLabels(labels)
              }

              // 3. Slot landmarks into Left/Right canonical channels
              // Note: Raw result.landmarks [0, 1] are used (unaffected by UI mirror)
              const slottedFrame = slotMediaPipeResult(result.landmarks, result.handedness, timestamp)

              // 4. Append to 60-frame rolling FIFO buffer
              frameBufferRef.current.push(slottedFrame)
              if (frameBufferRef.current.length > BUFFER_MAX_FRAMES) {
                frameBufferRef.current.shift()
              }

              const currentBufferLength = frameBufferRef.current.length
              if (prevBufferCountRef.current !== currentBufferLength) {
                prevBufferCountRef.current = currentBufferLength
                setBufferCount(currentBufferLength)
              }

              // 5. Inference Scheduling
              if (currentBufferLength < BUFFER_MAX_FRAMES) {
                // Still accumulating the initial 60-frame baseline window (~2.0 seconds)
                if (count === 0) {
                  setInferenceStatus('waiting_hands')
                } else {
                  setInferenceStatus('collecting_frames')
                }
              } else if (classifier) {
                // Buffer is full: evaluate every 15 frames (~250-500 ms interval)
                strideCounterRef.current += 1

                if (strideCounterRef.current >= INFERENCE_STRIDE_FRAMES) {
                  strideCounterRef.current = 0

                  // Compute 758-dim feature vector over the current 60-frame window
                  const features = computeVideoFeatureVector(frameBufferRef.current)

                  // Evaluate 49-class classifier through rejection gates
                  const prediction = classifySign(features, classifier, {
                    presenceThreshold: DEFAULT_PRESENCE_THRESHOLD,
                    motionThreshold: DEFAULT_MOTION_THRESHOLD,
                    confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
                    applyRejectionGates: true,
                  })

                  // Feed into Temporal Debouncer
                  const debounced = debouncerRef.current.update(prediction)
                  setLatestPrediction(prediction)

                  if (prediction.status === 'rejected') {
                    if (prediction.rejectionReason === 'presence_too_low') {
                      setInferenceStatus('waiting_hands')
                    } else if (prediction.rejectionReason === 'motion_stationary') {
                      setInferenceStatus('motion_stationary')
                    } else if (prediction.rejectionReason === 'confidence_low') {
                      setInferenceStatus('confidence_low')
                    } else if (prediction.rejectionReason === 'static_presence_low') {
                      setInferenceStatus('static_presence_low')
                    } else if (prediction.rejectionReason === 'static_pose_jitter') {
                      setInferenceStatus('static_pose_jitter')
                    } else if (prediction.rejectionReason === 'idle_hand_resting') {
                      setInferenceStatus('idle_hand_resting')
                    } else if (prediction.rejectionReason === 'static_confidence_ambiguous') {
                      setInferenceStatus('static_confidence_ambiguous')
                    }

                    // A pause or rejection resets the last-committed tracker, allowing
                    // the user to naturally sign the exact same word again after resting
                    lastCommittedLabelRef.current = null
                  } else {
                    // Gates 1, 2, and 3 passed!
                    if (debounced.isConfirmed && debounced.confirmedLabel) {
                      setInferenceStatus('sign_recognized')

                      const cleanWord = formatClassLabel(debounced.confirmedLabel)
                      const nowStr = new Date().toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })

                      setConfirmedSign({
                        label: debounced.confirmedLabel,
                        cleanWord,
                        confidence: prediction.confidence,
                        timestamp: nowStr,
                      })

                      // Rising-edge trigger: commit to transcript only on the exact cycle where debounce threshold is reached
                      if (debounced.consecutiveCount === debounced.requiredCount) {
                        setTranscript((prev) => (prev ? `${prev} ${cleanWord}` : cleanWord))
                        lastCommittedLabelRef.current = debounced.confirmedLabel

                        setRecentHistory((prev) => [
                          {
                            label: debounced.confirmedLabel!,
                            cleanWord,
                            confidence: prediction.confidence,
                            timestamp: nowStr,
                          },
                          ...prev.slice(0, 7), // retain last 8 confirmed signs
                        ])
                      }
                    } else {
                      // Actively evaluating gesture candidate (streak 1/2)
                      setInferenceStatus('recognizing')
                    }
                  }
                }
              }

              // Compute FPS telemetry
              frameCountRef.current += 1
              if (lastFpsUpdateRef.current === 0) {
                lastFpsUpdateRef.current = now
              } else if (now - lastFpsUpdateRef.current >= 1000) {
                setFps(Math.round((frameCountRef.current * 1000) / (now - lastFpsUpdateRef.current)))
                frameCountRef.current = 0
                lastFpsUpdateRef.current = now
              }
            } catch (detectErr) {
              console.error('Frame detection / inference error:', detectErr)
            }
          }
        }

        animFrameIdRef.current = requestAnimationFrame(runDetection)
      }

      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current)
      }
      animFrameIdRef.current = requestAnimationFrame(runDetection)
    } catch (err: unknown) {
      console.error('Camera access error:', err)
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setCameraError(
            'Camera permission was denied. Please allow camera access in your browser settings to continue.'
          )
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setCameraError('No video camera was detected on this device.')
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          setCameraError('Your camera is currently in use by another application.')
        } else {
          setCameraError(`Camera error: ${err.message || 'Unable to open video stream.'}`)
        }
      } else {
        setCameraError('Failed to initialize camera. Please check device permissions.')
      }
      stopCamera()
    } finally {
      setIsCameraLoading(false)
    }
  }

  // Ensure camera and detection loops are stopped when unmounting
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [stopCamera])

  // Clear translation output
  const handleClearOutput = () => {
    setTranscript('')
    setRecentHistory([])
    lastCommittedLabelRef.current = null
  }

  // Copy recognized text to clipboard
  const handleCopyText = async () => {
    if (!transcript) return
    try {
      await navigator.clipboard.writeText(transcript)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Clipboard copy failed:', err)
    }
  }

  // Explicitly save recognized translation to backend history
  const handleSaveTranslation = async () => {
    const cleanOutput = transcript.trim()
    if (!cleanOutput || isSavingTranslation) return
    setIsSavingTranslation(true)
    setSaveFeedback(null)
    try {
      await historyApi.create({
        source_language: 'ISL',
        translation_direction: 'sign_to_text',
        input_text: null,
        recognized_output: cleanOutput,
        confidence: confirmedSign?.confidence ?? latestPrediction?.confidence ?? null,
        recognition_path:
          latestPrediction?.recognitionPath && latestPrediction.recognitionPath !== 'none'
            ? latestPrediction.recognitionPath
            : null,
      })
      setSaveFeedback({ type: 'success', message: 'Saved to Translation History!' })
      setTimeout(() => setSaveFeedback(null), 3500)
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to save translation.'
      setSaveFeedback({ type: 'error', message: msg })
    } finally {
      setIsSavingTranslation(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Sign to Text Workspace
            </h1>
            {isCameraActive ? (
              <Badge variant="emerald" size="sm" dot>
                Camera Active
              </Badge>
            ) : cameraError ? (
              <Badge variant="amber" size="sm" dot>
                Permission Required
              </Badge>
            ) : (
              <Badge variant="slate" size="sm" dot>
                Camera Standby
              </Badge>
            )}
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Real-time hand tracking and 49-class Indian Sign Language isolated gesture classification.
          </p>
        </div>

        {/* Global Pipeline Health Status */}
        <div className="flex flex-wrap items-center gap-2">
          {/* MediaPipe Landmarker Status */}
          {isLandmarkerLoading ? (
            <Badge variant="amber" size="sm" dot className="font-mono text-xs">
              Loading Landmarker...
            </Badge>
          ) : isLandmarkerReady ? (
            <Badge variant="cyan" size="sm" dot className="font-mono text-xs">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              MediaPipe Online
            </Badge>
          ) : (
            <Badge variant="amber" size="sm" className="font-mono text-xs">
              Landmarker Offline
            </Badge>
          )}

          {/* Classifier Model Status */}
          {isClassifierLoading ? (
            <Badge variant="amber" size="sm" dot className="font-mono text-xs">
              Loading ISL Model...
            </Badge>
          ) : isClassifierReady ? (
            <Badge variant="emerald" size="sm" dot className="font-mono text-xs">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
              49-Class Model Active
            </Badge>
          ) : (
            <Badge variant="amber" size="sm" className="font-mono text-xs">
              ISL Model Offline
            </Badge>
          )}
        </div>
      </div>

      {/* Main Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Webcam HUD & MediaPipe Canvas Overlay (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="relative rounded-3xl p-1 bg-gradient-to-b from-cyan-500/20 via-slate-800/20 to-transparent border border-cyan-500/20 shadow-2xl">
            <div className="rounded-[22px] bg-[#070c1d] overflow-hidden backdrop-blur-xl border border-white/5 flex flex-col">
              {/* Top Viewport Header Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 px-3 sm:px-4 py-2.5 sm:py-3 bg-[#0a122c]/90 border-b border-white/5">
                <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-slate-300">
                  <Video className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span className="truncate">WEBCAM &amp; HAND MESH</span>
                  {isCameraActive && (
                    <span className="text-[11px] text-emerald-400 flex items-center gap-1 ml-0.5 font-mono">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      LIVE ({fps > 0 ? `${fps} FPS` : 'STREAMING'})
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {isCameraActive && (
                    <button
                      type="button"
                      onClick={() => setIsMirrored(!isMirrored)}
                      className={`text-xs px-3 py-1.5 min-h-[36px] sm:min-h-[32px] rounded-lg border transition-all cursor-pointer ${
                        isMirrored
                          ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}
                      title="Mirror camera horizontal orientation"
                    >
                      Mirror {isMirrored ? 'ON' : 'OFF'}
                    </button>
                  )}
                  <span className="hidden sm:inline-flex text-[10px] font-mono text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-white/5">
                    UP TO 2 HANDS
                  </span>
                </div>
              </div>

              {/* Video & Canvas Overlay Viewport (Portrait-friendly 3:4 on mobile, 4:3 tablet, 16:10 desktop) */}
              <div className="relative aspect-[3/4] sm:aspect-[4/3] lg:aspect-[16/10] max-h-[65vh] lg:max-h-none bg-[#040714] overflow-hidden flex items-center justify-center">
                {/* Subtle Grid Backdrop */}
                <div className="absolute inset-0 bg-grid-pattern opacity-25 pointer-events-none" />

                {/* Live Webcam Stream Video */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover transition-opacity duration-300 ${
                    isCameraActive ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'
                  } ${isMirrored ? 'scale-x-[-1]' : ''}`}
                />

                {/* Real-time MediaPipe Hand Landmarks Canvas Overlay */}
                <canvas
                  ref={canvasRef}
                  className={`absolute inset-0 w-full h-full object-cover pointer-events-none ${
                    isCameraActive ? 'opacity-100' : 'opacity-0'
                  }`}
                />

                {/* HUD Overlay Elements when camera is active */}
                {isCameraActive && (
                  <>
                    <div className="hidden sm:block absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2 border-cyan-400/80 pointer-events-none" />
                    <div className="hidden sm:block absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2 border-cyan-400/80 pointer-events-none" />
                    <div className="hidden sm:block absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2 border-cyan-400/80 pointer-events-none" />
                    <div className="hidden sm:block absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2 border-cyan-400/80 pointer-events-none" />

                    {/* Live Hand Detection Status Pill in Viewport */}
                    <div className="absolute top-2.5 sm:top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 sm:gap-2 bg-[#05091c]/85 border border-cyan-500/30 rounded-xl px-2.5 sm:px-3 py-1 sm:py-1.5 max-w-[92%] shadow-md backdrop-blur-md">
                      <Hand className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                      <span className="text-[11px] sm:text-xs font-mono font-medium text-white truncate">
                        {detectedHandsCount === 0 ? (
                          <span className="text-slate-400">Position hands in view</span>
                        ) : (
                          <span className="text-cyan-300">
                            {detectedHandsCount} {detectedHandsCount === 1 ? 'Hand' : 'Hands'} ({detectedHandsCount * 21} pts)
                          </span>
                        )}
                      </span>
                    </div>

                    {/* Bottom Left Privacy & Processing Badge */}
                    <div className="absolute bottom-3 left-3 sm:bottom-4 sm:left-4 z-10 hidden sm:flex items-center gap-2 bg-[#05091c]/90 border border-cyan-500/30 rounded-xl px-3 py-1.5 shadow-md backdrop-blur-md">
                      <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="text-[11px] font-mono text-slate-300">
                        100% In-Browser Inference
                      </span>
                    </div>

                    {/* Bottom Right Buffer Counter Badge */}
                    <div className="absolute bottom-2.5 right-2.5 sm:bottom-4 sm:right-4 z-10 flex items-center gap-1.5 bg-[#05091c]/85 border border-cyan-500/30 rounded-xl px-2.5 py-1 sm:px-3 sm:py-1.5 shadow-md backdrop-blur-md">
                      <Layers className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-cyan-400 shrink-0" />
                      <span className="text-[10px] sm:text-[11px] font-mono text-slate-300">
                        {bufferCount}/{BUFFER_MAX_FRAMES}
                      </span>
                    </div>
                  </>
                )}

                {/* Camera Standby Viewport State */}
                {!isCameraActive && !cameraError && (
                  <div className="text-center p-6 space-y-4 max-w-sm">
                    <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mx-auto shadow-inner">
                      <Camera className="w-8 h-8" />
                    </div>
                    <div className="space-y-1">
                      <h2 className="text-lg font-bold text-white tracking-wide">
                        Webcam Standby
                      </h2>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Start your camera to begin real-time hand landmark extraction and 49-class ISL gesture translation.
                      </p>
                    </div>
                  </div>
                )}

                {/* Camera Error State */}
                {cameraError && !isCameraLoading && (
                  <div className="relative z-10 text-center p-6 space-y-4 max-w-md">
                    <div className="w-14 h-14 rounded-2xl bg-amber-500/15 border border-amber-500/40 flex items-center justify-center text-amber-400 mx-auto">
                      <AlertCircle className="w-7 h-7" />
                    </div>
                    <div className="space-y-1">
                      <h2 className="text-base font-bold text-white">Camera Access Notice</h2>
                      <p className="text-xs text-amber-200/90 leading-relaxed bg-amber-950/40 border border-amber-500/20 p-3 rounded-xl">
                        {cameraError}
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={startCamera}
                      icon={<RefreshCw className="w-3.5 h-3.5" />}
                    >
                      Retry Camera Connection
                    </Button>
                  </div>
                )}
              </div>

              {/* Bottom Control Action Bar */}
              <div className="p-3 sm:p-4 bg-[#080d22] border-t border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2.5 w-full sm:w-auto">
                  {!isCameraActive ? (
                    <Button
                      variant="glow"
                      size="md"
                      onClick={startCamera}
                      disabled={isCameraLoading || isLandmarkerLoading || isClassifierLoading}
                      icon={<Camera className="w-4 h-4" />}
                      className="w-full sm:w-auto min-h-[44px]"
                    >
                      {isCameraLoading ? 'Starting Camera...' : 'Start Camera'}
                    </Button>
                  ) : (
                    <Button
                      variant="secondary"
                      size="md"
                      onClick={stopCamera}
                      className="w-full sm:w-auto min-h-[44px] border-rose-500/40 text-rose-300 hover:bg-rose-950/30 hover:border-rose-400"
                      icon={<CameraOff className="w-4 h-4" />}
                    >
                      Stop Camera
                    </Button>
                  )}

                  {isCameraActive && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsMirrored(!isMirrored)}
                      icon={<Maximize2 className="w-3.5 h-3.5" />}
                      className="min-h-[40px]"
                    >
                      Flip Horizontal
                    </Button>
                  )}
                </div>

                <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
                  <span className="text-slate-500">Telemetry:</span>
                  {isCameraActive ? (
                    <span className="text-cyan-400">
                      {detectedHandsCount > 0 ? `${detectedHandsCount} Hand(s) Active` : 'Scanning...'}
                    </span>
                  ) : (
                    <span className="text-slate-500">Standby</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Model Loading / Error Notices */}
          {(landmarkerError || classifierError) && (
            <div className="p-4 rounded-2xl bg-rose-950/30 border border-rose-500/30 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="text-xs font-semibold text-rose-300">Model Initialization Error</div>
                <div className="text-xs text-rose-200/80 leading-relaxed">
                  {landmarkerError || classifierError}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setIsLandmarkerLoading(true)
                    setIsClassifierLoading(true)
                    setLandmarkerError(null)
                    setClassifierError(null)
                    setRetryCount((c) => c + 1)
                  }}
                  className="mt-2 text-xs py-1"
                  icon={<RefreshCw className="w-3 h-3" />}
                >
                  Retry Loading Models
                </Button>
              </div>
            </div>
          )}

          {/* Real-time Hand Detection Telemetry Card */}
          <div className="p-5 rounded-2xl bg-[#070c1d] border border-white/5 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-white/5">
              <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider font-mono">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>Detection & Temporal Buffer Telemetry</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowDebugTelemetry((prev) => !prev)}
                  className="flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono text-cyan-300 hover:text-white bg-cyan-950/40 hover:bg-cyan-900/50 rounded-md border border-cyan-500/30 transition-colors"
                  title="Toggle Adaptive Motion Gate diagnostics"
                >
                  <Cpu className="w-3 h-3 text-cyan-400" />
                  <span>{showDebugTelemetry ? 'Hide Diagnostics' : 'Gating Diagnostics'}</span>
                </button>
                <Badge variant={detectedHandsCount > 0 ? 'cyan' : 'slate'} size="sm" className="font-mono">
                  {detectedHandsCount > 0 ? `${detectedHandsCount * 21} Points Tracked` : 'No Hands Detected'}
                </Badge>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Tracking Capacity</div>
                <div className="text-sm font-bold text-white font-mono">2 Hands (Max)</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Temporal Window</div>
                <div className="text-sm font-bold text-cyan-300 font-mono">
                  {bufferCount} / {BUFFER_MAX_FRAMES} frames
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Inference Stride</div>
                <div className="text-sm font-bold text-white font-mono">15 frames (~250ms)</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Identified Side</div>
                <div className="text-sm font-bold text-violet-300 font-mono truncate">
                  {handLabels.length > 0 ? handLabels.join(', ') : 'None'}
                </div>
              </div>
            </div>

            {/* Adaptive Motion Gate Debug Telemetry Grid */}
            {showDebugTelemetry && (
              <div className="mt-3 pt-3 border-t border-white/10 space-y-2.5 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-cyan-300 text-[11px] font-bold uppercase tracking-wider">
                    <Layers className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Adaptive Motion Gate Diagnostics</span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      latestPrediction?.status === 'accepted'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    }`}
                  >
                    {latestPrediction?.status === 'accepted' ? 'GATE: ACCEPTED' : 'GATE: REJECTED'}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                  {/* 1. Current Recognition Path */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Recognition Path</div>
                    <div className="font-bold text-white">
                      {latestPrediction?.recognitionPath === 'static_path' ? (
                        <span className="text-violet-300">Static (Path B)</span>
                      ) : latestPrediction?.recognitionPath === 'dynamic_path' ? (
                        <span className="text-cyan-300">Dynamic (Path A)</span>
                      ) : (
                        <span className="text-slate-400">None / Standby</span>
                      )}
                    </div>
                  </div>

                  {/* 2. Predicted Sign */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Predicted Sign</div>
                    <div
                      className="font-bold text-white truncate"
                      title={latestPrediction?.predictedLabel || latestPrediction?.topCandidates[0]?.label || 'None'}
                    >
                      {latestPrediction?.predictedLabel ? (
                        <span className="text-emerald-300">{formatClassLabel(latestPrediction.predictedLabel)}</span>
                      ) : latestPrediction?.topCandidates[0]?.label ? (
                        <span className="text-slate-400">{formatClassLabel(latestPrediction.topCandidates[0].label)}</span>
                      ) : (
                        'None'
                      )}
                    </div>
                  </div>

                  {/* 3. Confidence & Margin */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Confidence / Margin</div>
                    <div className="font-bold text-white">
                      {latestPrediction ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : '0.0%'}
                      <span className="text-[10px] text-slate-400 ml-1">
                        (Δ{latestPrediction?.gatingMetrics ? (latestPrediction.gatingMetrics.margin * 100).toFixed(1) : '0'}%)
                      </span>
                    </div>
                  </div>

                  {/* 4. Motion Energy */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Motion Energy (v)</div>
                    <div className="font-bold text-white">
                      <span
                        className={
                          (latestPrediction?.gatingMetrics?.motionEnergy ?? 0) >= 0.05
                            ? 'text-cyan-300'
                            : 'text-violet-300'
                        }
                      >
                        {(latestPrediction?.gatingMetrics?.motionEnergy ?? 0).toFixed(3)}
                      </span>
                      <span className="text-[10px] text-slate-500 ml-1">/ 0.050</span>
                    </div>
                  </div>

                  {/* 5. Hand Presence */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Hand Presence (L/R)</div>
                    <div className="font-bold text-white">
                      L: {(latestPrediction?.gatingMetrics?.presenceLeft ?? 0).toFixed(2)} | R:{' '}
                      {(latestPrediction?.gatingMetrics?.presenceRight ?? 0).toFixed(2)}
                    </div>
                  </div>

                  {/* 6. Pose Stability */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Pose Stability (σ)</div>
                    <div className="font-bold text-white">
                      <span
                        className={
                          (latestPrediction?.gatingMetrics?.spatialStd ?? 0) <= 0.055
                            ? 'text-emerald-300'
                            : 'text-rose-300'
                        }
                      >
                        {(latestPrediction?.gatingMetrics?.spatialStd ?? 0).toFixed(3)}
                      </span>
                      <span className="text-[10px] text-slate-500 ml-1">max 0.055</span>
                    </div>
                  </div>

                  {/* 7. Signing-Space Position */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Position (Centroid Y)</div>
                    <div className="font-bold text-white">
                      <span
                        className={
                          (latestPrediction?.gatingMetrics?.yCentroid ?? 1.0) <= 0.75
                            ? 'text-emerald-300'
                            : 'text-rose-300'
                        }
                      >
                        {(latestPrediction?.gatingMetrics?.yCentroid ?? 1.0).toFixed(2)}
                      </span>
                      <span className="text-[10px] text-slate-500 ml-1">max 0.75</span>
                    </div>
                  </div>

                  {/* 8. Acceptance Status & Rejection Reason */}
                  <div className="p-2 rounded-lg bg-slate-950/70 border border-white/5 space-y-0.5">
                    <div className="text-[10px] text-slate-500 uppercase">Rejection Reason</div>
                    <div className="font-bold truncate" title={latestPrediction?.rejectionReason || 'None (Accepted)'}>
                      {latestPrediction?.rejectionReason ? (
                        <span className="text-amber-300">{latestPrediction.rejectionReason}</span>
                      ) : (
                        <span className="text-emerald-400">None (Accepted)</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Output Panel & Positioning Guide (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Translation Output Card */}
          <div className="rounded-3xl p-1 bg-gradient-to-b from-cyan-500/20 via-violet-500/10 to-transparent border border-cyan-500/20 shadow-xl">
            <div className="rounded-[22px] bg-[#070c1d] p-5 sm:p-6 space-y-4">
              {/* Header */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-white/5">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400 shrink-0" />
                  <h2 className="text-base font-bold text-white tracking-wide">
                    Translation Output
                  </h2>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {transcript && (
                    <>
                      <button
                        type="button"
                        onClick={() => void handleSaveTranslation()}
                        disabled={isSavingTranslation}
                        className="flex items-center gap-1.5 px-3 py-1.5 min-h-[38px] text-xs text-cyan-300 hover:text-slate-950 bg-cyan-500/15 hover:bg-cyan-400 rounded-lg border border-cyan-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer font-medium"
                        title="Save translation to your history"
                      >
                        {isSavingTranslation ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            <span>Saving...</span>
                          </>
                        ) : (
                          <>
                            <Save className="w-3.5 h-3.5" />
                            <span>Save</span>
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={handleCopyText}
                        className="flex items-center gap-1 px-2.5 py-1 text-xs text-slate-300 hover:text-white bg-slate-800/80 hover:bg-slate-700/80 rounded-lg border border-white/10 transition-colors"
                        title="Copy text"
                      >
                        {copied ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    onClick={handleClearOutput}
                    disabled={!transcript && recentHistory.length === 0}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs text-slate-400 hover:text-rose-400 bg-slate-800/50 hover:bg-rose-950/30 rounded-lg border border-white/5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Clear translated output"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear Output</span>
                  </button>
                </div>
              </div>

              {saveFeedback && (
                <div
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs border ${
                    saveFeedback.type === 'success'
                      ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-200'
                      : 'bg-rose-950/40 border-rose-500/30 text-rose-200'
                  }`}
                >
                  {saveFeedback.type === 'success' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  ) : (
                    <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                  )}
                  <span>{saveFeedback.message}</span>
                </div>
              )}

              {/* Dynamic Rejection State & Recognition Status Banner */}
              <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900/80 border border-white/5 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-cyan-400" />
                    Recognition Status:
                  </span>
                  {latestPrediction?.recognitionPath === 'static_path' ? (
                    <span className="px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300 border border-violet-500/30 text-[10px]">
                      Path B (Static)
                    </span>
                  ) : latestPrediction?.recognitionPath === 'dynamic_path' ? (
                    <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-[10px]">
                      Path A (Dynamic)
                    </span>
                  ) : null}
                </div>
                {inferenceStatus === 'standby' && (
                  <span className="text-slate-400">Camera Standby</span>
                )}
                {inferenceStatus === 'waiting_hands' && (
                  <span className="text-amber-400 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    Waiting for Hands
                  </span>
                )}
                {inferenceStatus === 'collecting_frames' && (
                  <span className="text-cyan-300 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                    Buffering: {bufferCount}/{BUFFER_MAX_FRAMES} frames
                  </span>
                )}
                {inferenceStatus === 'motion_stationary' && (
                  <span className="text-blue-300 flex items-center gap-1">
                    <HelpCircle className="w-3.5 h-3.5 text-blue-400" />
                    Stationary (Move hands to sign)
                  </span>
                )}
                {inferenceStatus === 'confidence_low' && (
                  <span className="text-purple-300 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5 text-purple-400" />
                    Uncertain Sign (&lt; 45% score)
                  </span>
                )}
                {inferenceStatus === 'static_presence_low' && (
                  <span className="text-amber-300 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                    Partial Tracking (Low Presence)
                  </span>
                )}
                {inferenceStatus === 'static_pose_jitter' && (
                  <span className="text-amber-300 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                    Unstable Pose (Hold Hand Steady)
                  </span>
                )}
                {inferenceStatus === 'idle_hand_resting' && (
                  <span className="text-blue-300 flex items-center gap-1">
                    <HelpCircle className="w-3.5 h-3.5 text-blue-400" />
                    Hands Resting (Idle Position)
                  </span>
                )}
                {inferenceStatus === 'static_confidence_ambiguous' && (
                  <span className="text-purple-300 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5 text-purple-400" />
                    Uncertain Static Pose (&lt; 50% or Low Margin)
                  </span>
                )}
                {inferenceStatus === 'recognizing' && (
                  <span className="text-cyan-400 flex items-center gap-1 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                    Evaluating Candidate...
                  </span>
                )}
                {inferenceStatus === 'sign_recognized' && (
                  <span className="text-emerald-400 flex items-center gap-1 font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Sign Confirmed
                  </span>
                )}
              </div>

              {/* Output Content Area */}
              <div className="min-h-[140px] rounded-2xl bg-[#040714] border border-white/5 p-4 flex flex-col justify-between space-y-4">
                {transcript ? (
                  <div className="space-y-2">
                    <div className="text-xs uppercase font-mono text-cyan-400 tracking-wider">
                      Live Translated Sentence:
                    </div>
                    <div className="text-xl sm:text-2xl text-white font-bold tracking-tight leading-relaxed">
                      {transcript}
                    </div>
                  </div>
                ) : (
                  <div className="my-auto text-center space-y-2 py-4">
                    <p className="text-xs text-slate-400 font-mono">
                      [ Live ISL translation text will stream here as signs are recognized ]
                    </p>
                    <p className="text-[11px] text-slate-400">
                      Sustain an isolated gesture for 2 consecutive evaluation windows (60 frames) to commit it to the transcript.
                    </p>
                  </div>
                )}

                <div className="pt-3 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>Scope: 49 Isolated ISL Classes</span>
                  <span>
                    {detectedHandsCount > 0 ? (
                      <span className="text-cyan-400">{detectedHandsCount * 21} coordinates live</span>
                    ) : (
                      '0 landmarks'
                    )}
                  </span>
                </div>
              </div>

              {/* Active / Most Recently Confirmed Sign Card */}
              {confirmedSign && (
                <div className="p-4 rounded-2xl bg-gradient-to-r from-emerald-950/40 to-cyan-950/30 border border-emerald-500/30 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-emerald-300 text-xs font-semibold">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>Latest Confirmed Gesture</span>
                    </div>
                    <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-400" />
                      {confirmedSign.timestamp}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between">
                    <div className="text-2xl font-extrabold text-white tracking-tight">
                      {confirmedSign.cleanWord}
                    </div>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      {(confirmedSign.confidence * 100).toFixed(1)}% score (uncalibrated)
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400">
                    Dataset Class ID: <span className="text-cyan-300">{confirmedSign.label}</span>
                  </div>
                </div>
              )}

              {/* Top-3 Candidate Probabilities from Latest Evaluation Window */}
              {latestPrediction && latestPrediction.topCandidates.length > 0 && (
                <div className="p-4 rounded-2xl bg-slate-900/70 border border-white/5 space-y-2.5">
                  <div className="text-xs font-mono font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
                    <span>Top-3 Model Candidates (Latest Window)</span>
                    <span className="text-[10px] text-slate-500 lowercase">softmax decision scores</span>
                  </div>

                  <div className="space-y-2">
                    {latestPrediction.topCandidates.map((candidate, idx) => (
                      <div key={candidate.classIndex} className="space-y-1">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-200">
                            <span className="text-cyan-400 mr-1.5">{idx + 1}.</span>
                            {formatClassLabel(candidate.label)}{' '}
                            <span className="text-slate-500 text-[10px]">({candidate.label})</span>
                          </span>
                          <span className="text-slate-300 font-semibold">
                            {(candidate.score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              idx === 0 ? 'bg-cyan-400' : idx === 1 ? 'bg-violet-400' : 'bg-slate-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(0, candidate.score * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent Confirmed Gestures History */}
              {recentHistory.length > 0 && (
                <div className="p-4 rounded-2xl bg-slate-900/50 border border-white/5 space-y-2">
                  <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Recent Sign History
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {recentHistory.map((entry, i) => (
                      <span
                        key={i}
                        className="text-xs font-mono px-2 py-1 rounded-lg bg-slate-800/80 text-slate-200 border border-white/5 flex items-center gap-1.5"
                      >
                        <span className="text-cyan-400 font-semibold">{entry.cleanWord}</span>
                        <span className="text-[10px] text-slate-400">
                          {(entry.confidence * 100).toFixed(0)}%
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Honest Model Boundary Communication */}
              <div className="p-4 rounded-2xl bg-gradient-to-r from-violet-950/40 to-cyan-950/30 border border-violet-500/20 space-y-2">
                <div className="flex items-center gap-2 text-violet-300 text-xs font-semibold">
                  <Info className="w-4 h-4 shrink-0" />
                  <span>Model Architecture & Capability Boundary</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  <strong className="text-cyan-300">Model Scope:</strong> 49 isolated ISL sign classes from the Stage 5 baseline (74.31% held-out test accuracy). It classifies discrete spatio-temporal gestures spanning ~1.5–2.5 seconds.
                </p>
                <p className="text-xs text-slate-300 leading-relaxed">
                  <strong className="text-violet-300">Uncalibrated Scores:</strong> Softmax output scores represent linear decision boundary margins, not calibrated probabilities. Continuous sentence grammar and finger-spelling are not supported.
                </p>
              </div>
            </div>
          </div>

          {/* Positioning Guide (Requirement 8) */}
          <div className="rounded-3xl p-5 sm:p-6 bg-[#070c1d] border border-white/5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white">
              <Lightbulb className="w-4 h-4 text-cyan-400" />
              <span>How to Position for Optimal Recognition</span>
            </div>

            <ul className="space-y-3 text-xs text-slate-300">
              <li className="flex items-start gap-2.5">
                <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono flex items-center justify-center shrink-0 mt-0.5">
                  1
                </span>
                <div>
                  <span className="font-semibold text-slate-100">Optimal Illumination:</span> Face a
                  diffused light source. Avoid harsh backlights (like a bright window behind you) to
                  keep finger joints clearly visible.
                </div>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono flex items-center justify-center shrink-0 mt-0.5">
                  2
                </span>
                <div>
                  <span className="font-semibold text-slate-100">Torso & Face Framing:</span> Position
                  yourself so your head, shoulders, and chest stay within camera view.
                </div>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono flex items-center justify-center shrink-0 mt-0.5">
                  3
                </span>
                <div>
                  <span className="font-semibold text-slate-100">Gesture Duration:</span> Perform signs
                  with natural pacing (~1.5–2.0 seconds) so the 60-frame buffer captures the full trajectory.
                </div>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono flex items-center justify-center shrink-0 mt-0.5">
                  4
                </span>
                <div>
                  <span className="font-semibold text-slate-100">Dual-Hand Tracking:</span> MediaPipe
                  tracks up to two hands simultaneously (Hand 1 in Cyan, Hand 2 in Violet).
                </div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SignToTextPage
