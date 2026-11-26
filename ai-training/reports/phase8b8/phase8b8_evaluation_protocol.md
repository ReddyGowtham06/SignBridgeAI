# SIGNBRIDGE AI — PHASE 8B.8: INTERACTIVE HUMAN SIGNING EVALUATION PROTOCOL

> **Protocol Version:** 1.0.0  
> **Target Interface:** Development Evaluation Studio (`http://localhost:4173/app/evaluate` / `http://localhost:5173/app/evaluate`)  
> **Production Model Evaluated:** `frontend/public/models/sign_classifier.json` (M1 Softmax Logistic Regression)  
> **Hardware Requirements:** Host WebCam (`USB2.0 HD UVC WebCam`), minimum 720p 30 FPS  

---

## 1. Objective & Scope

This protocol defines the standardized procedure for conducting controlled, repeatable human sign language evaluation trials using the SIGNBRIDGE AI interactive evaluation studio. It ensures that all recorded attempts are empirically sound, reproducible, and compliant with strict data preservation and privacy guidelines.

---

## 2. Participant & Ethical Guidelines

1. **Non-Identifying Codes**: All human participants are assigned arbitrary, non-identifying participant codes (e.g., `P01`, `P02`).
2. **Informed Consent**: Participants must give explicit informed consent prior to participating.
3. **No Video Retention**: In compliance with Rule 8, no raw webcam video footage is retained on disk or uploaded to any training repository. Only structured numerical telemetry and categorical trial records are preserved.
4. **Authentic Signing**: Signs must be performed according to authentic Indian Sign Language (ISL) standards. If an evaluator or participant is uncertain of a sign's authentic kinematics, the attempt must be recorded as `invalid`.

---

## 3. Physical Environment & Hardware Setup

| Parameter | Recommended Specification | Acceptable Range |
| :--- | :--- | :--- |
| **Camera Model** | Built-in or external USB Webcam | `USB2.0 HD UVC WebCam` or equivalent |
| **Stream Resolution** | $1280 \times 720$ (720p) at 30 FPS | $640 \times 480$ to $1920 \times 1080$ |
| **Illumination** | Uniform indoor diffuse lighting | 300 to 500 lux; avoid direct backlighting |
| **Signer Distance** | $1.0\text{ meter}$ from camera lens | 0.8m to 1.2m |
| **Framing Viewport** | Head, shoulders, and both hands in view | Upper torso centered, hands unobstructed |
| **Background** | Clean, non-distracting background | Solid wall or neutral office environment |

---

## 4. Evaluation Studio Workflow

1. **Launch Evaluation Studio**:
   - Open browser at `http://localhost:4173/app/evaluate`.
   - Verify the presence of the amber development environment banner.
2. **Initialize Camera Stream**:
   - Click **"Launch Webcam"**. Confirm browser camera permissions.
   - Verify that 21-point hand skeletons render in real time (cyan for Hand 0, violet for Hand 1) and FPS is $\ge 25\text{ FPS}$.
3. **Select Target ISL Sign**:
   - For the Phase 8B.8 Pilot, select one of the 5 designated pilot classes:
     - `17.friend` (Index 8)
     - `21.good` (Index 13)
     - `29.nose` (Index 21)
     - `7.goodmorning` (Index 46)
     - `9.hello` (Index 48)
   - Read the displayed authentic ISL gestural motion and risk factor guidance.
4. **Execute Trial Attempt**:
   - Start with hands resting on lap or table (status: `waiting_hands`).
   - Click **"Start Attempt"** to initialize the trial timer.
   - Smoothly raise hands into the frame and perform the designated sign.
   - Maintain the final expressive configuration for **at least 1.5 seconds** (spanning at least two 15-frame inference strides).
   - Observe live telemetry: Top-1 prediction, Top-3 rankings, and Rejection gate notifications.
   - Lower hands to rest for 1.0 second to allow debouncer streak clearing.
5. **Classify Outcome & Log Record**:
   - Select the observed outcome:
     - `Correct Match`: The debounced transcript matched the ground-truth sign.
     - `Misclassified`: The system recognized a different sign class.
     - `Gate Rejection`: Hand was detected, but motion, confidence, or presence gate rejected the sample.
     - `Tracking Lost`: MediaPipe failed to detect or track the required hands.
     - `Invalid Sign`: Participant made an execution mistake or form error.
   - Enter diagnostic observations in the notes box.
   - Click **"Save Attempt Record"**.
6. **Repeat / Advance**:
   - Execute 5 attempts per sign before advancing to the next pilot class.
   - On completion, export the complete session log using **"Export Attempt Log (CSV)"**.
