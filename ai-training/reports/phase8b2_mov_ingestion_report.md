# SIGNBRIDGE AI — PHASE 8B.2: SAFE MOV VIDEO INGESTION REPORT

**Date**: 2026-11-01 | **Status**: 100% Verified Success | **Method**: Lossless Stream Copying (`-c copy`)

---

## Executive Summary

Phase 8B.2 executed a controlled, lossless container conversion of all 116 unprocessed QuickTime `.mov` videos in `D:\SignLang\sign project`. Prior to conversion, cryptographic SHA-256 deduplication identified 3 internal duplicate groups (7 files), from which exactly 1 deterministic canonical copy was selected per group.

- **Total Raw MOV Videos Audited**: 116
- **Unique Canonical Recordings Converted**: **112**
- **Duplicate Copies Excluded**: **4**
- **Conversion Success Rate**: **112 / 112 (100.0%)**
- **Conversion Failures**: **0**
- **Collisions with Existing MP4 Dataset**: **0**
- **Conversion Mode**: Lossless container stream copying (`-c copy -movflags +faststart`). No pixels were re-encoded.
- **Staging Directory**: `D:\SignBridgeAI\ai-training\data\staged_converted_movs`

---

## 1. MOV Deduplication Manifest

To prevent injecting intra-class duplicate redundancy into the dataset, cryptographic SHA-256 analysis isolated 3 duplicate groups. For each group, one deterministic canonical copy was selected, and identical duplicate copies were excluded from the conversion staging queue:

| Duplicate Group | SHA-256 Digest | Canonical File (Converted) | Excluded Duplicate Copies | Action |
| :--- | :--- | :--- | :--- | :--- |
| Group 1 | `3963f2f97f42d4cb...` | `12.afternoon/NachikethAfternoon1.mov` | `12.afternoon/NachikethAfternoon2.mov`<br>`12.afternoon/NachikethAfternoon3.mov` | `PRUNED_FROM_STAGING` |
| Group 2 | `15fcd9937510a574...` | `15.boy/NachikethBoy1.mov` | `15.boy/NachikethBoy2.mov` | `PRUNED_FROM_STAGING` |
| Group 3 | `301a506354a3bfb7...` | `35.student/Student2.mov` | `35.student/Student3.mov` | `PRUNED_FROM_STAGING` |

---

## 2. Conversion Quality & Fidelity Verification

Every single converted MP4 video was independently verified using OpenCV by probing stream geometry, frame boundaries, and frame counts:

- **Resolution Preservation**: 100% of converted files match source dimensions exactly (`1920x1080`).
- **Frame Count Exactness**: 100% of converted files match source frame counts exactly ($N_{\text{src}} == N_{\text{dst}}$).
- **Framerate Stability**: 100% of converted files match source framerates (29.97 / 30.00 FPS).
- **Full Frame Decodability**: Frame 0, middle frame ($T/2$), and final frame ($T-1$) decode with 100% success across all 112 videos.
- **Zero Dataset Collisions**: No output file SHA-256 collides with any of the 862 existing MP4 files in `D:\SignLang\sign project`.

---

## 3. Class-by-Class Converted Video Breakdown (All 39 Affected Classes)

| Class Name | Converted Videos Count | Sample Video Filename | Resolution | FPS | Avg Duration | Decodability | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `12.afternoon` | 1 | `NachikethAfternoon1.mp4` | `1920x1080` | 29.97 | 1.84s | 100% Verified | **READY** |
| `13.angry` | 3 | `NachikethAngry1.mp4` | `1920x1080` | 29.97 | 2.15s | 100% Verified | **READY** |
| `14.bad` | 3 | `NachikethBad1.mp4` | `1920x1080` | 29.97 | 1.51s | 100% Verified | **READY** |
| `15.boy` | 2 | `NachikethBoy1.mp4` | `1920x1080` | 29.97 | 2.35s | 100% Verified | **READY** |
| `16.eat` | 3 | `NachikethEat1.mp4` | `1920x1080` | 29.97 | 1.77s | 100% Verified | **READY** |
| `17.friend` | 3 | `NachikethFriend1.mp4` | `1920x1080` | 29.97 | 1.92s | 100% Verified | **READY** |
| `18.drinking` | 3 | `NachikethDrinking1.mp4` | `1920x1080` | 29.97 | 1.89s | 100% Verified | **READY** |
| `19.girl` | 3 | `NachikethGirl1.mp4` | `1920x1080` | 29.97 | 2.48s | 100% Verified | **READY** |
| `20.brother` | 3 | `NachikethBrother1.mp4` | `1920x1080` | 29.97 | 2.31s | 100% Verified | **READY** |
| `21.good` | 3 | `NachikethGood1.mp4` | `1920x1080` | 30.00 | 1.75s | 100% Verified | **READY** |
| `22.father` | 3 | `NachikethFather1.mp4` | `1920x1080` | 29.97 | 1.78s | 100% Verified | **READY** |
| `23.evening` | 3 | `NachikethEvening1.mp4` | `1920x1080` | 29.97 | 1.67s | 100% Verified | **READY** |
| `24.help` | 3 | `NachikethHelp1.mp4` | `1920x1080` | 29.97 | 1.95s | 100% Verified | **READY** |
| `25.mother` | 3 | `NachikethMother1.mp4` | `1920x1080` | 29.97 | 1.64s | 100% Verified | **READY** |
| `26.NAME` | 3 | `Name1.mp4` | `1920x1080` | 30.00 | 1.89s | 100% Verified | **READY** |
| `27.Night` | 3 | `Night1.mp4` | `1920x1080` | 29.97 | 2.26s | 100% Verified | **READY** |
| `28.Music` | 3 | `Music1.mp4` | `1920x1080` | 29.97 | 2.48s | 100% Verified | **READY** |
| `29.nose` | 3 | `Nose1.mp4` | `1920x1080` | 29.97 | 1.88s | 100% Verified | **READY** |
| `30.sleep` | 3 | `Sleep1.mp4` | `1920x1080` | 29.97 | 2.19s | 100% Verified | **READY** |
| `31.sit` | 3 | `Sit1.mp4` | `1920x1080` | 29.97 | 1.53s | 100% Verified | **READY** |
| `32.sorry` | 3 | `Sorry1.mp4` | `1920x1080` | 29.97 | 1.89s | 100% Verified | **READY** |
| `33.stand` | 3 | `Stand1.mp4` | `1920x1080` | 29.97 | 2.19s | 100% Verified | **READY** |
| `34.stop` | 3 | `Stop1.mp4` | `1920x1080` | 30.00 | 2.02s | 100% Verified | **READY** |
| `35.student` | 2 | `Student1.mp4` | `1920x1080` | 30.00 | 3.65s | 100% Verified | **READY** |
| `36.study` | 3 | `Study1.mp4` | `1920x1080` | 30.00 | 2.39s | 100% Verified | **READY** |
| `37.teacher` | 3 | `Teacher1.mp4` | `1920x1080` | 30.00 | 1.9s | 100% Verified | **READY** |
| `38.thankyou` | 3 | `ThankYou1.mp4` | `1920x1080` | 30.00 | 2.09s | 100% Verified | **READY** |
| `39.today` | 3 | `Today1.mp4` | `1920x1080` | 30.00 | 1.51s | 100% Verified | **READY** |
| `40.tommorow` | 3 | `Tomorrow1.mp4` | `1920x1080` | 30.00 | 1.78s | 100% Verified | **READY** |
| `41.welcome` | 3 | `Welcome1.mp4` | `1920x1080` | 30.00 | 2.3s | 100% Verified | **READY** |
| `42.work` | 3 | `Work1.mp4` | `1920x1080` | 30.00 | 1.61s | 100% Verified | **READY** |
| `43.yesterday` | 3 | `Yesterday1.mp4` | `1920x1080` | 30.00 | 1.62s | 100% Verified | **READY** |
| `44.teeth` | 3 | `Teeth1.mp4` | `1920x1080` | 30.00 | 2.05s | 100% Verified | **READY** |
| `45.hand` | 3 | `Hand1.mp4` | `1920x1080` | 30.00 | 1.53s | 100% Verified | **READY** |
| `46.write` | 3 | `Write1.mp4` | `1920x1080` | 30.00 | 1.64s | 100% Verified | **READY** |
| `47.umberlla` | 3 | `Umbrella1.mp4` | `1920x1080` | 30.00 | 1.93s | 100% Verified | **READY** |
| `48.ring` | 3 | `Ring1.mp4` | `1920x1080` | 30.00 | 2.02s | 100% Verified | **READY** |
| `5.dog` | 2 | `NachikethDog1.mp4` | `1920x1080` | 30.00 | 3.75s | 100% Verified | **READY** |
| `50.power` | 3 | `Power1.mp4` | `1920x1080` | 30.00 | 2.3s | 100% Verified | **READY** |

---

## 4. Specific Verification: `17.friend` Ingestion

The 3 Nachiketh FRIEND videos were converted and verified:

| Source MOV | Converted MP4 | Size | Resolution | FPS | Frames | Duration | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NachikethFriend1.mov` | `NachikethFriend1.mp4` | 981,874 B | `1920x1080` | 29.97 | 41 | 1.37s | **VERIFIED** |
| `NachikethFriend2.mov` | `NachikethFriend2.mp4` | 2,143,323 B | `1920x1080` | 29.97 | 76 | 2.54s | **VERIFIED** |
| `NachikethFriend3.mov` | `NachikethFriend3.mp4` | 1,518,742 B | `1920x1080` | 29.97 | 55 | 1.84s | **VERIFIED** |

---

## 5. Next Steps for Phase 8B.3

With 112 clean, lossless MP4 videos staged and verified in `ai-training/data/staged_converted_movs/`, the pipeline is primed for:
1. **Phase 8B.3: Controlled Landmark Extraction**: Extract MediaPipe hand landmarks from the 112 staged MP4 videos into a dedicated staging directory (`ai-training/data/staged_raw_landmarks/`), verifying presence and coordinate bounds.
2. **Phase 8B.4: Unified Dataset Assembly**: Merge the 112 verified landmark archives with the cleaned 862-sample baseline (after quarantining the 8 cross-class conflicting samples).

---
*Report generated automatically by Phase 8B.2 Safe MOV Ingestion Suite.*