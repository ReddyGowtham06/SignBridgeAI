import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Camera,
  CameraOff,
  Video,
  Trash2,
  Download,
  AlertTriangle,
  ShieldCheck,
  Activity,
  Hand,
  Clock,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Play,
  Square,
  ChevronRight,
  ChevronLeft,
  Filter,
  Sliders,
  Award,
} from 'lucide-react'
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
import {
  loadM5Model,
  classifySignM5,
  type LoadedM5Model,
} from '../../utils/m5Evaluator'
import {
  evaluateAdaptiveMotionGate,
  type AdaptiveGateResult,
} from '../../utils/adaptiveMotionGate'

// 21 standard HandLandmarker skeletal connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]

const BUFFER_MAX_FRAMES = 60
const INFERENCE_STRIDE_FRAMES = 15

// Authentic 49-Class Mapping from label_mapping.json
const ISL_CLASSES: Array<{ id: number; label: string; displayName: string; category: string; description: string }> = [
  { id: 0, label: '1.bird', displayName: 'Bird', category: 'Single-Hand', description: 'Index finger and thumb pinching open/close near mouth mimicking a beak' },
  { id: 1, label: '10.youareperfect', displayName: 'You Are Perfect', category: 'Compound', description: 'Point to person then form OK or thumbs-up gesture' },
  { id: 2, label: '11.monsoon', displayName: 'Monsoon', category: 'Dual-Hand', description: 'Both open hands moving downwards with fluttering fingers mimicking rain' },
  { id: 3, label: '12.afternoon', displayName: 'Afternoon', category: 'Single-Hand', description: 'Flat hand held diagonally over forearm representing sun past zenith' },
  { id: 4, label: '13.angry', displayName: 'Angry', category: 'Single-Hand', description: 'Clawed hand drawn up in front of face/chest with intense tension' },
  { id: 5, label: '14.bad', displayName: 'Bad', category: 'Single-Hand', description: 'Flat hand drawn from chin and turned palm downward decisively' },
  { id: 6, label: '15.boy', displayName: 'Boy', category: 'Single-Hand', description: 'Hand grabbing or tapping imaginary cap brim at forehead' },
  { id: 7, label: '16.eat', displayName: 'Eat', category: 'Facial-Proximity', description: 'Fingertips bunched together brought towards and tapping near mouth' },
  { id: 8, label: '17.friend', displayName: 'Friend', category: 'Two-Hand Occlusion', description: 'Both hands brought together in front of chest with index fingers interlocked' },
  { id: 9, label: '18.drinking', displayName: 'Drinking', category: 'Facial-Proximity', description: 'C-handshape tilted upwards toward mouth mimicking drinking from cup' },
  { id: 10, label: '19.girl', displayName: 'Girl', category: 'Single-Hand', description: 'Thumb or index finger stroking along jawline from ear to chin' },
  { id: 11, label: '2.black', displayName: 'Black', category: 'Facial-Proximity', description: 'Index finger pointing and stroking across eyebrow or hairline' },
  { id: 12, label: '20.brother', displayName: 'Brother', category: 'Dual-Hand', description: 'Boy sign (cap gesture) combined with index fingers held side-by-side' },
  { id: 13, label: '21.good', displayName: 'Good', category: 'Low-Motion Static', description: 'Firm thumbs-up gesture held upright in front of chest' },
  { id: 14, label: '22.father', displayName: 'Father', category: 'Single-Hand', description: 'Thumb tapping or resting on forehead with open fingers' },
  { id: 15, label: '23.evening', displayName: 'Evening', category: 'Single-Hand', description: 'Bent flat hand moving downward over wrist representing setting sun' },
  { id: 16, label: '24.help', displayName: 'Help', category: 'Dual-Hand', description: 'Closed fist with thumb up resting on flat palm of other hand, lifted upward' },
  { id: 17, label: '25.mother', displayName: 'Mother', category: 'Single-Hand', description: 'Thumb tapping chin or cheek with open fingers spread' },
  { id: 18, label: '26.NAME', displayName: 'Name', category: 'Dual-Hand', description: 'H-fingers (index and middle) tapping across each other twice' },
  { id: 19, label: '27.Night', displayName: 'Night', category: 'Dual-Hand', description: 'Bent right hand moving down over horizontal left forearm' },
  { id: 20, label: '28.Music', displayName: 'Music', category: 'Dual-Hand', description: 'Right hand waving back and forth over horizontal left forearm' },
  { id: 21, label: '29.nose', displayName: 'Nose', category: 'Facial-Proximity', description: 'Index finger clearly touching or pointing to the tip of the nose' },
  { id: 22, label: '3.cat', displayName: 'Cat', category: 'Single-Hand', description: 'Fingers pinching imaginary whiskers outwards from cheek' },
  { id: 23, label: '30.sleep', displayName: 'Sleep', category: 'Facial-Proximity', description: 'Open hand drawn down over face while fingers gently close together' },
  { id: 24, label: '31.sit', displayName: 'Sit', category: 'Dual-Hand', description: 'Two curved fingers of right hand hooked over two curved fingers of left hand' },
  { id: 25, label: '32.sorry', displayName: 'Sorry', category: 'Single-Hand', description: 'Closed fist rubbing in circular motion over sternum/chest' },
  { id: 26, label: '33.stand', displayName: 'Stand', category: 'Dual-Hand', description: 'Index and middle fingers of right hand inverted standing on flat left palm' },
  { id: 27, label: '34.stop', displayName: 'Stop', category: 'Low-Motion Static', description: 'Open palm facing forward held firm and upright' },
  { id: 28, label: '35.student', displayName: 'Student', category: 'Dual-Hand', description: 'Flat hand scooping information from palm up toward forehead' },
  { id: 29, label: '36.study', displayName: 'Study', category: 'Dual-Hand', description: 'Open fingers fluttering and pointing down toward flat left palm' },
  { id: 30, label: '37.teacher', displayName: 'Teacher', category: 'Dual-Hand', description: 'Both hands near temples pulling outward then moving downward' },
  { id: 31, label: '38.thankyou', displayName: 'Thank You', category: 'Single-Hand', description: 'Flat hand moving from chin forward and downward toward receiver' },
  { id: 32, label: '39.today', displayName: 'Today', category: 'Dual-Hand', description: 'Both hands in Y-shape or open palms dropping downward twice' },
  { id: 33, label: '4.cow', displayName: 'Cow', category: 'Single-Hand', description: 'Y-handshape with thumb touching temple mimicking cow horn' },
  { id: 34, label: '40.tommorow', displayName: 'Tomorrow', category: 'Single-Hand', description: 'Thumb on cheek twisting or moving forward past chin' },
  { id: 35, label: '41.welcome', displayName: 'Welcome', category: 'Dual-Hand', description: 'Both open palms swept smoothly inward toward body' },
  { id: 36, label: '42.work', displayName: 'Work', category: 'Two-Hand Occlusion', description: 'Heel of dominant fist repeatedly tapping base of non-dominant fist' },
  { id: 37, label: '43.yesterday', displayName: 'Yesterday', category: 'Single-Hand', description: 'Thumb or index finger moving backward along jawline toward ear' },
  { id: 38, label: '44.teeth', displayName: 'Teeth', category: 'Facial-Proximity', description: 'Index finger pointing across exposed teeth' },
  { id: 39, label: '45.hand', displayName: 'Hand', category: 'Low-Motion Static', description: 'Open hand held forward with palm exposed to camera' },
  { id: 40, label: '46.write', displayName: 'Write', category: 'Dual-Hand', description: 'Dominant hand mimicking holding a pen scribbling across flat left palm' },
  { id: 41, label: '47.umberlla', displayName: 'Umbrella', category: 'Two-Hand Occlusion', description: 'Two fists stacked vertically moving upward as if opening umbrella' },
  { id: 42, label: '48.ring', displayName: 'Ring', category: 'Dual-Hand', description: 'Index finger sliding imaginary ring onto ring finger of other hand' },
  { id: 43, label: '5.dog', displayName: 'Dog', category: 'Single-Hand', description: 'Snapping fingers twice or patting thigh' },
  { id: 44, label: '50.power', displayName: 'Power', category: 'Dual-Hand', description: 'Both hands clenched in strong biceps flexing posture' },
  { id: 45, label: '6.fish', displayName: 'Fish', category: 'Single-Hand', description: 'Flat hand swimming forward in undulating sine-wave motion' },
  { id: 46, label: '7.goodmorning', displayName: 'Good Morning', category: 'Compound', description: 'Good thumbs-up gesture transitioning into sunrise or morning greeting wave' },
  { id: 47, label: '8.grey', displayName: 'Grey', category: 'Single-Hand', description: 'Open fingers combing through hair at side of head' },
  { id: 48, label: '9.hello', displayName: 'Hello', category: 'Single-Hand', description: 'Open palm facing forward waving side to side near shoulder' },
]


const CATEGORIES = [
  'All',
  'Single-Hand',
  'Dual-Hand',
  'Facial-Proximity',
  'Two-Hand Occlusion',
  'Low-Motion Static',
  'Compound',
]

export interface TrialLogRecord {
  trialId: string;
  classId: number;
  classLabel: string;
  attemptNumber: number;
  participantCode: string;
  lightingCondition: string;
  cameraDistance: string;
  handPositioning: string;
  signingSpeed: string;
  handDetectionStatus: string;
  leftPresenceRatio: number;
  rightPresenceRatio: number;
  top1Prediction: string;
  top1Confidence: number;
  top3Candidates: string;
  m5Top1Prediction?: string;
  m5Top1Confidence?: number;
  modelsAgree?: boolean;
  adaptiveStatus?: string;
  adaptivePrediction?: string;
  adaptiveRejectionReason?: string;
  adaptivePath?: string;
  motionEnergy?: number;
  spatialStd?: number;
  yCentroid?: number;
  top1Margin?: number;
  executionVariation?: string;
  rejectionReason: string;
  debouncedTranscript: string;
  recognitionCorrectness: 'correct' | 'incorrect' | 'rejected' | 'undetected' | 'invalid';
  recognitionLatencyMs: number | null;
  evaluatorNotes: string;
  timestamp: string;
}

export const EvaluationStudioPage = () => {
  // Evaluation Workflow Selection
  const [selectedClassId, setSelectedClassId] = useState<number>(8) // default to 17.friend
  const [categoryFilter, setCategoryFilter] = useState<string>('All')
  const [participantCode, setParticipantCode] = useState<string>('P01')
  const [lightingCondition, setLightingCondition] = useState<string>('Normal Indoor (300-500 lux)')
  const [cameraDistance, setCameraDistance] = useState<string>('Standard (~1.0m)')
  const [handPositioning, setHandPositioning] = useState<string>('Centered')
  const [signingSpeed, setSigningSpeed] = useState<string>('Normal Tempo (1.5-2.5s)')
  const [executionVariation, setExecutionVariation] = useState<string>('Standard Hold')
  const [adaptiveGateResult, setAdaptiveGateResult] = useState<AdaptiveGateResult | null>(null)
  const [evaluatorNotes, setEvaluatorNotes] = useState<string>('')
  const [selectedOutcome, setSelectedOutcome] = useState<'correct' | 'incorrect' | 'rejected' | 'undetected' | 'invalid'>('correct')

  // Trial execution state
  const [isTrialActive, setIsTrialActive] = useState<boolean>(false)
  const [trialStartTime, setTrialStartTime] = useState<number | null>(null)
  const [trialElapsedMs, setTrialElapsedMs] = useState<number>(0)
  const [trialCountByClass, setTrialCountByClass] = useState<Record<number, number>>({})
  const [loggedTrials, setLoggedTrials] = useState<TrialLogRecord[]>([])

  // Camera & Pipeline state
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const [isCameraLoading, setIsCameraLoading] = useState<boolean>(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [fps, setFps] = useState<number>(0)
  const [detectedHandsCount, setDetectedHandsCount] = useState<number>(0)
  const [leftPresenceLive, setLeftPresenceLive] = useState<number>(0)
  const [rightPresenceLive, setRightPresenceLive] = useState<number>(0)
  const [bufferCount, setBufferCount] = useState<number>(0)
  const [inferenceStatus, setInferenceStatus] = useState<string>('standby')
  const [latestPrediction, setLatestPrediction] = useState<PredictionResult | null>(null)
  const [m5Prediction, setM5Prediction] = useState<PredictionResult | null>(null)
  const [debouncedCandidate, setDebouncedCandidate] = useState<string | null>(null)
  const [debouncedStreak, setDebouncedStreak] = useState<number>(0)
  const [confirmedLabel, setConfirmedLabel] = useState<string | null>(null)

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const classifierModelRef = useRef<LoadedModel | null>(null)
  const m5ModelRef = useRef<LoadedM5Model | null>(null)
  const debouncerRef = useRef<TemporalDebouncer>(new TemporalDebouncer({ minConsecutiveFrames: 2 }))
  const frameBufferRef = useRef<SlottedFrame[]>([])
  const strideCounterRef = useRef<number>(0)
  const animFrameIdRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const lastTimestampRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)
  const lastFpsUpdateRef = useRef<number>(0)
  const trialTimerIntervalRef = useRef<number | null>(null)

  const selectedClass = ISL_CLASSES.find((c) => c.id === selectedClassId) || ISL_CLASSES[0]
  const filteredClasses = categoryFilter === 'All'
    ? ISL_CLASSES
    : ISL_CLASSES.filter((c) => c.category === categoryFilter)

  // Model & Landmarker Initialization
  useEffect(() => {
    let isMounted = true

    async function initPipeline() {
      try {
        let vision
        try {
          vision = await FilesetResolver.forVisionTasks('/wasm')
        } catch {
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
        } catch {
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

        const [model, m5] = await Promise.all([
          loadModel('/models/sign_classifier.json'),
          loadM5Model('/models/m5_evaluator.json').catch((err) => {
            console.warn('M5 evaluator model not loaded:', err)
            return null
          }),
        ])

        if (isMounted) {
          landmarkerRef.current = landmarker
          classifierModelRef.current = model
          m5ModelRef.current = m5
        }
      } catch (err) {
        console.error('Evaluation pipeline init failure:', err)
      }
    }

    initPipeline()

    return () => {
      isMounted = false
      if (landmarkerRef.current) {
        landmarkerRef.current.close()
      }
    }
  }, [])

  // Draw overlay landmarks
  const drawOverlay = (
    canvas: HTMLCanvasElement,
    video: HTMLVideoElement,
    result: HandLandmarkerResult
  ) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!result.landmarks || result.landmarks.length === 0) return

    const { width, height } = canvas
    result.landmarks.forEach((hand, idx) => {
      const strokeColor = idx === 0 ? 'rgba(6, 182, 212, 0.9)' : 'rgba(168, 85, 247, 0.9)'
      const fillColor = idx === 0 ? '#22d3ee' : '#c084fc'

      ctx.save()
      ctx.lineWidth = 2.5
      ctx.strokeStyle = strokeColor
      for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = hand[s]
        const p2 = hand[e]
        if (!p1 || !p2) continue
        ctx.beginPath()
        ctx.moveTo((1 - p1.x) * width, p1.y * height)
        ctx.lineTo((1 - p2.x) * width, p2.y * height)
        ctx.stroke()
      }

      hand.forEach((p) => {
        ctx.beginPath()
        ctx.arc((1 - p.x) * width, p.y * height, 3.5, 0, 2 * Math.PI)
        ctx.fillStyle = fillColor
        ctx.fill()
      })
      ctx.restore()
    })
  }

  // Stop Camera
  const stopCamera = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current)
      animFrameIdRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setIsCameraActive(false)
    setIsCameraLoading(false)
    setFps(0)
    setDetectedHandsCount(0)
    frameBufferRef.current = []
    strideCounterRef.current = 0
    debouncerRef.current.reset()
    setBufferCount(0)
    setInferenceStatus('standby')
  }, [])

  // Start Camera
  const startCamera = async () => {
    setCameraError(null)
    setIsCameraLoading(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      setIsCameraActive(true)

      const runLoop = () => {
        const video = videoRef.current
        const canvas = canvasRef.current
        const landmarker = landmarkerRef.current
        const classifier = classifierModelRef.current

        if (video && canvas && landmarker && video.readyState >= 2 && !video.paused) {
          if (video.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = video.currentTime
            const now = performance.now()
            const timestamp = now > lastTimestampRef.current ? now : lastTimestampRef.current + 1
            lastTimestampRef.current = timestamp

            try {
              const result = landmarker.detectForVideo(video, timestamp)
              drawOverlay(canvas, video, result)

              const handsCount = result.landmarks?.length || 0
              setDetectedHandsCount(handsCount)

              const slotted = slotMediaPipeResult(result.landmarks, result.handedness, timestamp)
              frameBufferRef.current.push(slotted)
              if (frameBufferRef.current.length > BUFFER_MAX_FRAMES) {
                frameBufferRef.current.shift()
              }

              const bufLen = frameBufferRef.current.length
              setBufferCount(bufLen)

              // Compute presence telemetry across current buffer
              if (bufLen > 0) {
                const leftPres = frameBufferRef.current.filter((f) => f.landmarks[0] !== null).length / bufLen
                const rightPres = frameBufferRef.current.filter((f) => f.landmarks[1] !== null).length / bufLen
                setLeftPresenceLive(leftPres)
                setRightPresenceLive(rightPres)
              }

              if (bufLen < BUFFER_MAX_FRAMES) {
                setInferenceStatus(handsCount === 0 ? 'waiting_hands' : 'collecting_frames')
              } else if (classifier) {
                strideCounterRef.current += 1
                if (strideCounterRef.current >= INFERENCE_STRIDE_FRAMES) {
                  strideCounterRef.current = 0

                  const features = computeVideoFeatureVector(frameBufferRef.current)
                  const pred = classifySign(features, classifier, {
                    presenceThreshold: DEFAULT_PRESENCE_THRESHOLD,
                    motionThreshold: DEFAULT_MOTION_THRESHOLD,
                    confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
                    applyRejectionGates: true,
                  })

                  const adapt = evaluateAdaptiveMotionGate(features, classifier)
                  setAdaptiveGateResult(adapt)

                  if (m5ModelRef.current) {
                    const predM5 = classifySignM5(features, m5ModelRef.current, {
                      presenceThreshold: DEFAULT_PRESENCE_THRESHOLD,
                      motionThreshold: DEFAULT_MOTION_THRESHOLD,
                      confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
                      applyRejectionGates: true,
                    })
                    setM5Prediction(predM5)
                  }

                  const debounced = debouncerRef.current.update(pred)
                  setLatestPrediction(pred)
                  setDebouncedCandidate(debounced.candidateLabel)
                  setDebouncedStreak(debounced.consecutiveCount)

                  if (pred.status === 'rejected') {
                    setInferenceStatus(pred.rejectionReason || 'rejected')
                  } else if (debounced.isConfirmed && debounced.confirmedLabel) {
                    setInferenceStatus('sign_recognized')
                    setConfirmedLabel(debounced.confirmedLabel)
                  } else {
                    setInferenceStatus('recognizing')
                  }
                }
              }

              // FPS calculation
              frameCountRef.current += 1
              if (lastFpsUpdateRef.current === 0) {
                lastFpsUpdateRef.current = now
              } else if (now - lastFpsUpdateRef.current >= 1000) {
                setFps(Math.round((frameCountRef.current * 1000) / (now - lastFpsUpdateRef.current)))
                frameCountRef.current = 0
                lastFpsUpdateRef.current = now
              }
            } catch (loopErr) {
              console.error('Evaluation loop error:', loopErr)
            }
          }
        }
        animFrameIdRef.current = requestAnimationFrame(runLoop)
      }

      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current)
      animFrameIdRef.current = requestAnimationFrame(runLoop)
    } catch (err: unknown) {
      setCameraError(err instanceof Error ? err.message : 'Camera initialization failed.')
      stopCamera()
    } finally {
      setIsCameraLoading(false)
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
      if (trialTimerIntervalRef.current) clearInterval(trialTimerIntervalRef.current)
    }
  }, [stopCamera])

  // Start Trial
  const handleStartTrial = () => {
    setIsTrialActive(true)
    const t0 = performance.now()
    setTrialStartTime(t0)
    setTrialElapsedMs(0)
    setConfirmedLabel(null)
    debouncerRef.current.reset()

    if (trialTimerIntervalRef.current) clearInterval(trialTimerIntervalRef.current)
    trialTimerIntervalRef.current = window.setInterval(() => {
      setTrialElapsedMs(Math.round(performance.now() - t0))
    }, 50)
  }

  // Stop / Pause Trial
  const handleStopTrial = () => {
    setIsTrialActive(false)
    if (trialTimerIntervalRef.current) {
      clearInterval(trialTimerIntervalRef.current)
      trialTimerIntervalRef.current = null
    }
  }

  // Log Record
  const handleLogAttempt = () => {
    const currentClassTrials = (trialCountByClass[selectedClassId] || 0) + 1
    const trialId = `TR-${String(loggedTrials.length + 1).padStart(3, '0')}`

    const top3Str = latestPrediction?.topCandidates
      ? latestPrediction.topCandidates
          .slice(0, 3)
          .map((c) => `${c.label}:${c.score.toFixed(3)}`)
          .join('|')
      : 'none'

    const m5PredLabel = m5Prediction?.predictedLabel || 'none'
    const m5Conf = m5Prediction?.confidence || 0.0
    const m1PredLabel = latestPrediction?.predictedLabel || 'none'
    const modelsAgree = m1PredLabel !== 'none' && m1PredLabel === m5PredLabel

    const rec: TrialLogRecord = {
      trialId,
      classId: selectedClass.id,
      classLabel: selectedClass.label,
      attemptNumber: currentClassTrials,
      participantCode,
      lightingCondition,
      cameraDistance,
      handPositioning,
      signingSpeed,
      handDetectionStatus:
        detectedHandsCount >= 2 ? 'Both Hands' : detectedHandsCount === 1 ? 'Single Hand' : 'None',
      leftPresenceRatio: +leftPresenceLive.toFixed(3),
      rightPresenceRatio: +rightPresenceLive.toFixed(3),
      top1Prediction: latestPrediction?.predictedLabel || 'none',
      top1Confidence: latestPrediction?.confidence || 0.0,
      top3Candidates: top3Str,
      m5Top1Prediction: m5PredLabel,
      m5Top1Confidence: m5Conf,
      modelsAgree,
      adaptiveStatus: adaptiveGateResult?.adaptiveGate.status || 'none',
      adaptivePrediction: adaptiveGateResult?.adaptiveGate.predictedLabel || 'none',
      adaptiveRejectionReason: adaptiveGateResult?.adaptiveGate.rejectionReason || 'none',
      adaptivePath: adaptiveGateResult?.adaptiveGate.pathTaken || 'none',
      motionEnergy: adaptiveGateResult?.metrics.motionEnergy || 0.0,
      spatialStd: adaptiveGateResult?.metrics.spatialStd || 0.0,
      yCentroid: adaptiveGateResult?.metrics.yCentroid || 1.0,
      top1Margin: adaptiveGateResult?.metrics.margin || 0.0,
      executionVariation,
      rejectionReason: latestPrediction?.rejectionReason || 'none',
      debouncedTranscript: confirmedLabel || 'none',
      recognitionCorrectness: selectedOutcome,
      recognitionLatencyMs: trialStartTime ? trialElapsedMs : null,
      evaluatorNotes: evaluatorNotes.trim() || 'No specific anomalies observed',
      timestamp: new Date().toISOString(),
    }

    setLoggedTrials((prev) => [rec, ...prev])
    setTrialCountByClass((prev) => ({ ...prev, [selectedClassId]: currentClassTrials }))
    handleStopTrial()
    setEvaluatorNotes('')
  }

  // Export CSV
  const handleExportCSV = () => {
    if (loggedTrials.length === 0) {
      alert('No trial records logged yet.')
      return
    }

    const headers = [
      'trial_id',
      'class_id',
      'class_label',
      'attempt_number',
      'participant_code',
      'lighting_condition',
      'camera_distance',
      'hand_positioning',
      'signing_speed',
      'hand_detection_status',
      'left_presence_ratio',
      'right_presence_ratio',
      'top1_prediction',
      'top1_confidence',
      'top3_candidates',
      'm5_top1_prediction',
      'm5_top1_confidence',
      'models_agree',
      'adaptive_status',
      'adaptive_prediction',
      'adaptive_rejection_reason',
      'adaptive_path',
      'motion_energy',
      'spatial_std',
      'y_centroid',
      'top1_margin',
      'execution_variation',
      'rejection_reason',
      'debounced_transcript',
      'recognition_correctness',
      'recognition_latency_ms',
      'evaluator_notes',
      'timestamp',
    ]

    const csvRows = [headers.join(',')]
    loggedTrials.forEach((r) => {
      csvRows.push(
        [
          r.trialId,
          r.classId,
          r.classLabel,
          r.attemptNumber,
          `"${r.participantCode}"`,
          `"${r.lightingCondition}"`,
          `"${r.cameraDistance}"`,
          `"${r.handPositioning}"`,
          `"${r.signingSpeed}"`,
          `"${r.handDetectionStatus}"`,
          r.leftPresenceRatio,
          r.rightPresenceRatio,
          `"${r.top1Prediction}"`,
          r.top1Confidence.toFixed(4),
          `"${r.top3Candidates}"`,
          `"${r.m5Top1Prediction ?? 'none'}"`,
          (r.m5Top1Confidence ?? 0).toFixed(4),
          r.modelsAgree !== undefined ? (r.modelsAgree ? 'TRUE' : 'FALSE') : 'N/A',
          `"${r.adaptiveStatus ?? 'none'}"`,
          `"${r.adaptivePrediction ?? 'none'}"`,
          `"${r.adaptiveRejectionReason ?? 'none'}"`,
          `"${r.adaptivePath ?? 'none'}"`,
          (r.motionEnergy ?? 0).toFixed(4),
          (r.spatialStd ?? 0).toFixed(4),
          (r.yCentroid ?? 1.0).toFixed(4),
          (r.top1Margin ?? 0).toFixed(4),
          `"${r.executionVariation ?? 'Standard'}"`,
          `"${r.rejectionReason}"`,
          `"${r.debouncedTranscript}"`,
          r.recognitionCorrectness,
          r.recognitionLatencyMs ?? '',
          `"${r.evaluatorNotes.replace(/"/g, '""')}"`,
          r.timestamp,
        ].join(',')
      )
    })

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `phase8b9_live_trial_log_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Sequential class navigation
  const handlePrevClass = () => {
    const currentIdx = ISL_CLASSES.findIndex((c) => c.id === selectedClassId)
    const prevIdx = (currentIdx - 1 + ISL_CLASSES.length) % ISL_CLASSES.length
    setSelectedClassId(ISL_CLASSES[prevIdx].id)
    setConfirmedLabel(null)
    debouncerRef.current.reset()
  }

  const handleNextClass = () => {
    const currentIdx = ISL_CLASSES.findIndex((c) => c.id === selectedClassId)
    const nextIdx = (currentIdx + 1) % ISL_CLASSES.length
    setSelectedClassId(ISL_CLASSES[nextIdx].id)
    setConfirmedLabel(null)
    debouncerRef.current.reset()
  }



  // Summary Metrics computed from logged trials
  const totalCompleted = loggedTrials.length
  const correctCount = loggedTrials.filter((t) => t.recognitionCorrectness === 'correct').length
  const incorrectCount = loggedTrials.filter((t) => t.recognitionCorrectness === 'incorrect').length
  const rejectedCount = loggedTrials.filter((t) => t.recognitionCorrectness === 'rejected').length
  const undetectedCount = loggedTrials.filter((t) => t.recognitionCorrectness === 'undetected').length
  const invalidCount = loggedTrials.filter((t) => t.recognitionCorrectness === 'invalid').length
  const validDenominator = totalCompleted - invalidCount
  const liveAccuracyPercent = validDenominator > 0 ? ((correctCount / validDenominator) * 100).toFixed(1) : '0.0'

  return (
    <div className="space-y-6 pb-12">
      {/* DEVELOPMENT WARNING BANNER */}
      <div className="bg-amber-950/40 border-2 border-amber-500/50 rounded-2xl p-4 flex items-start gap-3 shadow-lg shadow-amber-950/30">
        <div className="p-2 rounded-xl bg-amber-500/20 text-amber-400 mt-0.5">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div className="flex-1 text-sm">
          <div className="flex items-center gap-2">
            <span className="font-bold text-amber-300 uppercase tracking-wider text-xs">
              Development Evaluation Environment — Phase 8B.9: Full 49-Class Human Webcam Baseline
            </span>
            <Badge variant="amber" size="sm">
              Non-Production Harness
            </Badge>
          </div>
          <p className="text-amber-200/80 mt-1 text-xs sm:text-sm">
            This interface is dedicated to conducting controlled, reproducible manual signing trials across all 49 ISL vocabulary classes.
            It evaluates the <strong>untouched production M1 champion model</strong> and live MediaPipe pipeline without altering thresholds or features.
          </p>
        </div>
      </div>

      {/* TOP HEADER & PROGRESS */}
      <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl p-6 shadow-xl flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-black tracking-wide text-white flex items-center gap-2">
              <Activity className="w-6 h-6 text-cyan-400" />
              Interactive Human Signing Studio — 49-Class Baseline
            </h1>
            <Badge variant="cyan" size="sm">
              M1 Model Verified
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Perform authentic ISL signs in front of your webcam, inspect live pipeline telemetry, and record trial logs across all 49 vocabulary classes.
          </p>
        </div>

        {/* 49-CLASS EVALUATION PROGRESS */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 bg-slate-950/80 border border-slate-800 rounded-xl p-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Award className="w-4 h-4 text-amber-400" /> 49-Class Coverage:
          </span>
          <div className="flex items-center gap-2">
            <Badge variant="cyan" size="sm">
              {Object.keys(trialCountByClass).filter((k) => (trialCountByClass[Number(k)] || 0) >= 3).length} / 49 Classes Done
            </Badge>
            <span className="text-xs font-mono text-slate-400">
              ({totalCompleted} / 147 min trials)
            </span>
          </div>
        </div>
      </div>

      {/* MAIN TWO-COLUMN WORKFLOW */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN: LIVE WEBCAM & TELEMETRY (7 COLS) */}
        <div className="lg:col-span-7 space-y-6">
          {/* CAMERA VIEWPORT CARD */}
          <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl overflow-hidden shadow-2xl relative">
            {/* Viewport Top Bar */}
            <div className="px-4 py-3 bg-slate-950/80 border-b border-cyan-500/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    isCameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                  }`}
                />
                <span className="text-xs font-mono font-semibold uppercase text-slate-300">
                  {isCameraActive ? 'Live Webcam Stream (720p)' : 'Camera Off'}
                </span>
                {fps > 0 && (
                  <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-500/30">
                    {fps} FPS
                  </span>
                )}
              </div>

              {/* Inference Status Pill */}
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-400 font-mono">Status:</span>
                <span
                  className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider border ${
                    inferenceStatus === 'sign_recognized'
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 animate-pulse'
                      : inferenceStatus === 'recognizing'
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      : inferenceStatus === 'motion_stationary'
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                      : inferenceStatus === 'confidence_low'
                      ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {inferenceStatus.replace(/_/g, ' ')}
                </span>
              </div>
            </div>

            {/* Video + Canvas Stack */}
            <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
              <video
                ref={videoRef}
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              />

              {/* Inactive Overlay */}
              {!isCameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/90 text-slate-400 p-6 text-center">
                  <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shadow-lg">
                    <Camera className="w-7 h-7" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">Camera Standby</h3>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm">
                      Initialize your local webcam to start recording manual evaluation trials.
                    </p>
                  </div>
                  <Button
                    variant="glow"
                    size="sm"
                    icon={<Video className="w-4 h-4" />}
                    onClick={startCamera}
                    disabled={isCameraLoading}
                  >
                    {isCameraLoading ? 'Accessing Camera...' : 'Launch Webcam'}
                  </Button>
                </div>
              )}

              {/* Active Trial Live Timer Overlay */}
              {isCameraActive && isTrialActive && (
                <div className="absolute top-3 left-3 bg-slate-950/90 border border-cyan-500/40 rounded-xl px-3 py-1.5 flex items-center gap-2 backdrop-blur-md shadow-lg">
                  <Clock className="w-4 h-4 text-cyan-400 animate-spin" />
                  <span className="text-xs font-mono font-bold text-white">
                    {(trialElapsedMs / 1000).toFixed(1)}s elapsed
                  </span>
                </div>
              )}
            </div>

            {/* Viewport Controls Bar */}
            <div className="p-3 bg-slate-950/90 border-t border-cyan-500/10 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                {isCameraActive ? (
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-rose-500/40 text-rose-400 hover:bg-rose-950/30"
                    icon={<CameraOff className="w-4 h-4" />}
                    onClick={stopCamera}
                  >
                    Stop Stream
                  </Button>
                ) : (
                  <Button variant="glow" size="sm" icon={<Camera className="w-4 h-4" />} onClick={startCamera}>
                    Start Stream
                  </Button>
                )}

                {isCameraActive && (
                  <>
                    {!isTrialActive ? (
                      <Button
                        variant="primary"
                        size="sm"
                        icon={<Play className="w-4 h-4" />}
                        onClick={handleStartTrial}
                      >
                        Start Attempt #{((trialCountByClass[selectedClassId] || 0) + 1)}
                      </Button>
                    ) : (
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={<Square className="w-4 h-4" />}
                        onClick={handleStopTrial}
                      >
                        Freeze Attempt
                      </Button>
                    )}
                  </>
                )}
              </div>

              {/* Live Detection Telemetry Pills */}
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="bg-slate-900 px-2 py-1 rounded border border-slate-700 text-slate-300">
                  Buffer: <strong className="text-cyan-400">{bufferCount}</strong>/60
                </span>
                <span className="bg-slate-900 px-2 py-1 rounded border border-slate-700 text-slate-300">
                  Hands: <strong className="text-cyan-400">{detectedHandsCount}</strong> (L:{' '}
                  {(leftPresenceLive * 100).toFixed(0)}%, R: {(rightPresenceLive * 100).toFixed(0)}%)
                </span>
              </div>
            </div>

            {/* Camera Error Banner */}
            {cameraError && (
              <div className="p-3 bg-rose-950/80 border-t border-rose-500/30 text-xs text-rose-300 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{cameraError}</span>
              </div>
            )}
          </div>

          {/* REAL-TIME MODEL TELEMETRY CARD */}
          <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                <Sliders className="w-4 h-4" /> Live Dual-Model Telemetry (M1 vs M5)
              </h3>
              {latestPrediction && m5Prediction && latestPrediction.predictedLabel !== 'none' && (
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    latestPrediction.predictedLabel === m5Prediction.predictedLabel
                      ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30'
                      : 'bg-amber-950/60 text-amber-400 border-amber-500/30'
                  }`}
                >
                  {latestPrediction.predictedLabel === m5Prediction.predictedLabel
                    ? '✓ Models Agree'
                    : '⚠ Disagreement'}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* M1 Production Output */}
              <div className="bg-slate-950/70 border border-cyan-500/30 rounded-xl p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-cyan-400 font-bold">M1 Champion (Linear)</span>
                  <Badge variant="cyan" size="sm">Prod</Badge>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-extrabold text-white truncate max-w-[130px]">
                    {latestPrediction?.predictedLabel || 'None'}
                  </span>
                  {latestPrediction?.confidence && (
                    <span className="text-xs font-mono font-bold text-cyan-400">
                      {(latestPrediction.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {latestPrediction?.rejectionReason && (
                  <div className="text-[10px] text-amber-400 font-mono truncate">
                    Gate: {latestPrediction.rejectionReason}
                  </div>
                )}
              </div>

              {/* M5 Experimental Output */}
              <div className="bg-slate-950/70 border border-purple-500/30 rounded-xl p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-purple-400 font-bold">M5 Candidate (MLP)</span>
                  <Badge variant="slate" size="sm">Exp</Badge>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-extrabold text-white truncate max-w-[130px]">
                    {m5Prediction?.predictedLabel || 'None'}
                  </span>
                  {m5Prediction?.confidence && (
                    <span className="text-xs font-mono font-bold text-purple-400">
                      {(m5Prediction.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {m5Prediction?.rejectionReason && (
                  <div className="text-[10px] text-amber-400 font-mono truncate">
                    Gate: {m5Prediction.rejectionReason}
                  </div>
                )}
              </div>

              {/* Confirmed Debounced Card */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Debounced Text</span>
                  <span className="text-[10px] font-mono text-slate-400">Streak: {debouncedStreak}/2</span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-extrabold text-emerald-400 truncate max-w-[150px]">
                    {confirmedLabel || 'Awaiting...'}
                  </span>
                </div>
                <div className="text-[10px] text-slate-500 truncate">
                  Cand: {debouncedCandidate || 'none'}
                </div>
              </div>
            </div>

            {/* Top-3 Candidate Ranking Bar */}
            {latestPrediction?.topCandidates && latestPrediction.topCandidates.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                  Top-3 Ranked Predictions:
                </span>
                <div className="space-y-1">
                  {latestPrediction.topCandidates.slice(0, 3).map((cand, i) => (
                    <div key={cand.classIndex} className="flex items-center gap-2 text-xs">
                      <span className="w-4 text-slate-500 font-mono">#{i + 1}</span>
                      <span className="w-36 font-semibold text-slate-200 truncate">{cand.label}</span>
                      <div className="flex-1 bg-slate-900 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-cyan-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(100, cand.score * 100)}%` }}
                        />
                      </div>
                      <span className="text-[11px] font-mono text-slate-400 w-12 text-right">
                        {(cand.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: TRIAL RECORDING DECK (5 COLS) */}
        <div className="lg:col-span-5 space-y-6">
          {/* TARGET CLASS SELECTION CARD */}
          <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                <Award className="w-4 h-4" /> Target Evaluation Sign
              </h3>
              <Badge variant="slate" size="sm">
                Class {selectedClass.id} of 49
              </Badge>
            </div>

            {/* Category Filter Pills */}
            <div>
              <label className="block text-[11px] font-medium text-slate-400 mb-1.5 flex items-center gap-1">
                <Filter className="w-3 h-3 text-cyan-400" /> Filter Movement Category:
              </label>
              <div className="flex flex-wrap gap-1">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setCategoryFilter(cat)}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-all ${
                      categoryFilter === cat
                        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400 font-bold'
                        : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Class Dropdown */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Select ISL Target Class ({filteredClasses.length} of 49):
              </label>
              <select
                value={selectedClassId}
                onChange={(e) => {
                  setSelectedClassId(Number(e.target.value))
                  setConfirmedLabel(null)
                  debouncerRef.current.reset()
                }}
                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white font-medium focus:border-cyan-400 focus:outline-none"
              >
                {filteredClasses.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label} ({c.displayName}) — [{c.category}]
                  </option>
                ))}
              </select>
            </div>

            {/* Ground Truth Details Card */}
            <div className="bg-slate-950/80 border border-cyan-500/10 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-black text-white">{selectedClass.displayName}</span>
                <Badge variant="cyan" size="sm">
                  {selectedClass.category}
                </Badge>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                <strong>Authentic ISL Motion:</strong> {selectedClass.description}
              </p>
              <div className="text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800 flex items-center justify-between">
                <button
                  type="button"
                  onClick={handlePrevClass}
                  className="text-cyan-400 hover:underline flex items-center gap-0.5"
                >
                  <ChevronLeft className="w-3 h-3" /> Prev Sign
                </button>
                <span>Attempts: <strong>{trialCountByClass[selectedClassId] || 0}</strong> / 3 target</span>
                <button
                  type="button"
                  onClick={handleNextClass}
                  className="text-cyan-400 hover:underline flex items-center gap-0.5"
                >
                  Next Sign <ChevronRight className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>

          {/* TRIAL LOGGING FORM */}
          <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl p-5 shadow-xl space-y-4">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" /> Trial Observation & Record
            </h3>

            {/* Trial Conditions Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Participant Code:
                </label>
                <input
                  type="text"
                  value={participantCode}
                  onChange={(e) => setParticipantCode(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white font-mono text-xs focus:border-cyan-400"
                />
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Lighting Condition:
                </label>
                <select
                  value={lightingCondition}
                  onChange={(e) => setLightingCondition(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-xs focus:border-cyan-400"
                >
                  <option value="Normal Indoor (300-500 lux)">Normal Indoor</option>
                  <option value="Low Ambient (<100 lux)">Low Light (&lt;100 lux)</option>
                  <option value="Direct / Harsh Lighting">Direct / Harsh</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Camera Distance:
                </label>
                <select
                  value={cameraDistance}
                  onChange={(e) => setCameraDistance(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-xs focus:border-cyan-400"
                >
                  <option value="Standard (~1.0m)">Standard (~1.0m)</option>
                  <option value="Close (~0.5m)">Close (~0.5m)</option>
                  <option value="Far (~1.5m)">Far (~1.5m)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Hand Positioning:
                </label>
                <select
                  value={handPositioning}
                  onChange={(e) => setHandPositioning(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-xs focus:border-cyan-400"
                >
                  <option value="Centered">Centered</option>
                  <option value="Slightly Off-Center">Slightly Off-Center</option>
                  <option value="Periphery / Borderline">Periphery / Borderline</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Signing Speed:
                </label>
                <select
                  value={signingSpeed}
                  onChange={(e) => setSigningSpeed(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-xs focus:border-cyan-400"
                >
                  <option value="Normal Tempo (1.5-2.5s)">Normal Tempo</option>
                  <option value="Fast / Flick (<1.0s)">Fast / Quick</option>
                  <option value="Sustained Hold (>3.0s)">Sustained Hold</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">
                  Execution Variation:
                </label>
                <select
                  value={executionVariation}
                  onChange={(e) => setExecutionVariation(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-xs focus:border-cyan-400"
                >
                  <option value="Standard Hold">Standard Hold</option>
                  <option value="Sustained Hold (>3s)">Sustained Hold (&gt;3s)</option>
                  <option value="Natural Motion">Natural Motion</option>
                  <option value="Varied Position">Varied Position</option>
                  <option value="Desk Resting">Desk Resting</option>
                  <option value="Lap Resting">Lap Resting</option>
                  <option value="Casual Signing Space">Casual Signing Space</option>
                  <option value="Natural Fidgeting">Natural Fidgeting</option>
                  <option value="Frame Entry/Exit">Frame Entry/Exit</option>
                </select>
              </div>
            </div>

            {/* Trial Outcome Selector */}
            <div>
              <label className="text-[11px] font-semibold text-slate-300 block mb-1.5 uppercase tracking-wider">
                Observed Trial Outcome:
              </label>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setSelectedOutcome('correct')}
                  className={`p-2 rounded-lg font-semibold border flex items-center gap-1.5 transition-all ${
                    selectedOutcome === 'correct'
                      ? 'bg-emerald-950/80 border-emerald-400 text-emerald-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Correct Match
                </button>

                <button
                  type="button"
                  onClick={() => setSelectedOutcome('incorrect')}
                  className={`p-2 rounded-lg font-semibold border flex items-center gap-1.5 transition-all ${
                    selectedOutcome === 'incorrect'
                      ? 'bg-rose-950/80 border-rose-400 text-rose-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  <XCircle className="w-4 h-4 text-rose-400" /> Misclassified
                </button>

                <button
                  type="button"
                  onClick={() => setSelectedOutcome('rejected')}
                  className={`p-2 rounded-lg font-semibold border flex items-center gap-1.5 transition-all ${
                    selectedOutcome === 'rejected'
                      ? 'bg-amber-950/80 border-amber-400 text-amber-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  <AlertTriangle className="w-4 h-4 text-amber-400" /> Gate Rejection
                </button>

                <button
                  type="button"
                  onClick={() => setSelectedOutcome('undetected')}
                  className={`p-2 rounded-lg font-semibold border flex items-center gap-1.5 transition-all ${
                    selectedOutcome === 'undetected'
                      ? 'bg-purple-950/80 border-purple-400 text-purple-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  <Hand className="w-4 h-4 text-purple-400" /> Tracking Lost
                </button>
              </div>
            </div>

            {/* Evaluator Notes */}
            <div>
              <label className="text-[11px] font-medium text-slate-400 block mb-1">
                Diagnostic Observations / Anomaly Notes:
              </label>
              <textarea
                value={evaluatorNotes}
                onChange={(e) => setEvaluatorNotes(e.target.value)}
                placeholder="e.g. MediaPipe dropped right hand when fingers interlocked; Gate 2 motion stationary triggered..."
                rows={2}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white placeholder-slate-500 focus:border-cyan-400"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2 pt-1">
              <Button
                variant="glow"
                size="sm"
                className="flex-1"
                icon={<CheckCircle2 className="w-4 h-4" />}
                onClick={handleLogAttempt}
              >
                Save Attempt Record
              </Button>
              <Button
                variant="ghost"
                size="sm"
                icon={<RotateCcw className="w-4 h-4" />}
                onClick={() => {
                  setConfirmedLabel(null)
                  debouncerRef.current.reset()
                  handleStartTrial()
                }}
              >
                Repeat
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* SESSION SUMMARY & TRIAL AUDIT LOG */}
      <div className="bg-[#070c1d] border border-cyan-500/20 rounded-2xl p-6 shadow-xl space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cyan-500/10 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-cyan-400" />
              Pilot Evaluation Session Telemetry
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Empirical trials logged during this active human evaluation session.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="secondary"
              size="sm"
              icon={<Download className="w-4 h-4" />}
              onClick={handleExportCSV}
              disabled={loggedTrials.length === 0}
            >
              Export Attempt Log ({loggedTrials.length})
            </Button>
            {loggedTrials.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="border-rose-500/40 text-rose-400 hover:bg-rose-950/30"
                icon={<Trash2 className="w-4 h-4" />}
                onClick={() => {
                  if (confirm('Clear all logged trial records from this browser session?')) {
                    setLoggedTrials([])
                    setTrialCountByClass({})
                  }
                }}
              >
                Clear Log
              </Button>
            )}
          </div>
        </div>

        {/* METRICS SUMMARY TILES */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-slate-400 uppercase">Completed</span>
            <div className="text-xl font-black text-white mt-0.5">{totalCompleted}</div>
          </div>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-emerald-400 uppercase">Correct</span>
            <div className="text-xl font-black text-emerald-400 mt-0.5">{correctCount}</div>
          </div>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-rose-400 uppercase">Incorrect</span>
            <div className="text-xl font-black text-rose-400 mt-0.5">{incorrectCount}</div>
          </div>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-amber-400 uppercase">Rejected</span>
            <div className="text-xl font-black text-amber-400 mt-0.5">{rejectedCount}</div>
          </div>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-purple-400 uppercase">Undetected</span>
            <div className="text-xl font-black text-purple-400 mt-0.5">{undetectedCount}</div>
          </div>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center">
            <span className="text-[10px] font-mono text-cyan-400 uppercase">Live Accuracy</span>
            <div className="text-xl font-black text-cyan-400 mt-0.5">{liveAccuracyPercent}%</div>
          </div>
        </div>

        {/* TRIAL LOG TABLE */}
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950 text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">ID</th>
                <th className="py-2.5 px-3">Target Sign</th>
                <th className="py-2.5 px-3">Attempt</th>
                <th className="py-2.5 px-3">Top-1 Prediction</th>
                <th className="py-2.5 px-3">Top-1 Conf</th>
                <th className="py-2.5 px-3">Debounced Output</th>
                <th className="py-2.5 px-3">Outcome</th>
                <th className="py-2.5 px-3">Latency</th>
                <th className="py-2.5 px-3">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
              {loggedTrials.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-slate-500 font-sans">
                    No trials recorded yet in this session. Launch camera and click &quot;Start Attempt&quot; above to log evaluations.
                  </td>
                </tr>
              ) : (
                loggedTrials.map((t) => (
                  <tr key={t.trialId} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-2 px-3 font-bold text-slate-400">{t.trialId}</td>
                    <td className="py-2 px-3 text-white font-semibold">{t.classLabel}</td>
                    <td className="py-2 px-3">#{t.attemptNumber}</td>
                    <td className="py-2 px-3 text-cyan-300">{t.top1Prediction}</td>
                    <td className="py-2 px-3">{(t.top1Confidence * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-emerald-400">{t.debouncedTranscript}</td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          t.recognitionCorrectness === 'correct'
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/30'
                            : t.recognitionCorrectness === 'incorrect'
                            ? 'bg-rose-950 text-rose-300 border border-rose-500/30'
                            : t.recognitionCorrectness === 'rejected'
                            ? 'bg-amber-950 text-amber-300 border border-amber-500/30'
                            : 'bg-purple-950 text-purple-300 border border-purple-500/30'
                        }`}
                      >
                        {t.recognitionCorrectness}
                      </span>
                    </td>
                    <td className="py-2 px-3">{t.recognitionLatencyMs ? `${t.recognitionLatencyMs} ms` : '—'}</td>
                    <td className="py-2 px-3 text-slate-400 truncate max-w-xs">{t.evaluatorNotes}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
export default EvaluationStudioPage
