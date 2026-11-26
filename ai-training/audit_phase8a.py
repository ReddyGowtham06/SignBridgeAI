import os
import sys
import re
import json
import time
import hashlib
from collections import defaultdict, Counter
import numpy as np
import cv2

RAW_VIDEO_DIR = r"D:\SignLang\sign project"
RAW_LANDMARKS_DIR = r"D:\SignBridgeAI\ai-training\data\raw_landmarks"
PROCESSED_DIR = r"D:\SignBridgeAI\ai-training\data\processed"
REPORTS_DIR = r"D:\SignBridgeAI\ai-training\reports"

def compute_file_hash(filepath, block_size=65536):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_signer_from_filename(filename):
    fname = os.path.splitext(filename)[0]
    fname_lower = fname.lower()
    
    if fname.startswith("0403"):
        return "Signer_0403"
    elif fname_lower.startswith("krishnaraj"):
        return "Signer_Krishnaraj"
    elif fname_lower.startswith("manideep"):
        return "Signer_Manideep"
    elif fname_lower.startswith("nachiketh"):
        return "Signer_Nachiketh"
    elif fname.startswith("M_") or fname.startswith("M-") or (fname.startswith("M") and len(fname) > 1 and fname[1].isupper()):
        return "Signer_Batch_M"
    elif fname.startswith("S_") or fname.startswith("S-") or (fname.startswith("S") and len(fname) > 1 and fname[1].isupper() and not fname.startswith("Sleep") and not fname.startswith("Sit") and not fname.startswith("Stop") and not fname.startswith("Student") and not fname.startswith("Study") and not fname.startswith("Sorry")):
        return "Signer_Batch_S"
    elif fname.startswith("P_") or fname.startswith("P-") or (fname.startswith("P") and len(fname) > 1 and fname[1].isupper() and not fname.startswith("Power")):
        return "Signer_Batch_P"
    elif " -" in fname or " -" in fname:
        # e.g. "Bird -1", "Friend -2"
        return "Signer_Batch_Hyphen"
    elif re.search(r"^[A-Za-z]+\s*\d+$", fname):
        # e.g. "Bird1", "Bird 2", "Monsoon 1"
        return "Signer_Batch_Numeric"
    elif "_" in fname:
        return "Signer_Batch_Underscore"
    else:
        return "Signer_Unclassified"

def audit_dataset():
    print("=== STARTING PHASE 8A DATASET AUDIT ===")
    start_time = time.time()
    
    # -------------------------------------------------------------
    # 1. DATASET INVENTORY
    # -------------------------------------------------------------
    print("\n--- 1. Auditing Dataset Inventory ---")
    class_folders = sorted(os.listdir(RAW_VIDEO_DIR))
    class_folders = [f for f in class_folders if os.path.isdir(os.path.join(RAW_VIDEO_DIR, f))]
    
    total_classes = len(class_folders)
    print(f"Total class folders found: {total_classes}")
    
    # Check numeric continuity
    class_numbers = []
    class_names_clean = []
    for cf in class_folders:
        parts = cf.split(".", 1)
        if len(parts) == 2 and parts[0].isdigit():
            class_numbers.append(int(parts[0]))
            class_names_clean.append(parts[1])
        else:
            print(f"Warning: Non-standard class folder name: {cf}")
            
    missing_numbers = [i for i in range(1, 51) if i not in class_numbers]
    print(f"Class numbers range: {min(class_numbers)} to {max(class_numbers)}")
    print(f"Missing numbers in sequence 1..50: {missing_numbers}")
    
    # Inventory scan
    video_records = []
    hash_to_files = defaultdict(list)
    class_inventory = {}
    total_raw_videos = 0
    ext_counter = Counter()
    
    for cf in class_folders:
        cdir = os.path.join(RAW_VIDEO_DIR, cf)
        files = sorted(os.listdir(cdir))
        mp4_files = [f for f in files if f.lower().endswith(".mp4")]
        mov_files = [f for f in files if f.lower().endswith(".mov")]
        other_files = [f for f in files if not f.lower().endswith(".mp4") and not f.lower().endswith(".mov")]
        
        class_inventory[cf] = {
            "total_files": len(files),
            "mp4_count": len(mp4_files),
            "mov_count": len(mov_files),
            "other_count": len(other_files),
            "files": files
        }
        total_raw_videos += len(files)
        
        for f in files:
            fpath = os.path.join(cdir, f)
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext] += 1
            fsize = os.path.getsize(fpath)
            fhash = compute_file_hash(fpath)
            signer = extract_signer_from_filename(f)
            
            hash_to_files[fhash].append({
                "class_name": cf,
                "filename": f,
                "filepath": fpath,
                "size_bytes": fsize
            })
            
            video_records.append({
                "class_name": cf,
                "filename": f,
                "filepath": fpath,
                "extension": ext,
                "size_bytes": fsize,
                "sha256": fhash,
                "signer_group": signer
            })

    print(f"Total raw video files: {total_raw_videos}")
    print(f"Extensions: {dict(ext_counter)}")
    
    # Exact Duplicates
    exact_duplicate_groups = []
    intra_class_duplicates = []
    inter_class_duplicates = []
    
    for fhash, flist in hash_to_files.items():
        if len(flist) > 1:
            group = {
                "sha256": fhash,
                "size_bytes": flist[0]["size_bytes"],
                "count": len(flist),
                "files": flist
            }
            exact_duplicate_groups.append(group)
            
            classes = set(item["class_name"] for item in flist)
            if len(classes) == 1:
                intra_class_duplicates.append(group)
            else:
                inter_class_duplicates.append(group)
                
    print(f"Total exact duplicate file groups: {len(exact_duplicate_groups)}")
    print(f"  Intra-class duplicates (same sign class): {len(intra_class_duplicates)}")
    print(f"  Inter-class duplicates (different sign classes): {len(inter_class_duplicates)}")
    for item in inter_class_duplicates:
        print(f"    WARNING: Inter-class duplicate: {[f['class_name'] + '/' + f['filename'] for f in item['files']]}")

    # -------------------------------------------------------------
    # 2. VIDEO QUALITY AUDIT
    # -------------------------------------------------------------
    print("\n--- 2. Auditing Video Quality (decoding, resolution, fps, blur) ---")
    resolutions = Counter()
    durations = []
    fps_list = []
    sharpness_scores = []
    brightness_scores = []
    corrupted_videos = []
    
    # Process each video (sample 5 frames for speed and precision)
    for idx, vrec in enumerate(video_records):
        cap = cv2.VideoCapture(vrec["filepath"])
        if not cap.isOpened():
            corrupted_videos.append(vrec["filepath"])
            vrec["decode_status"] = "CANNOT_OPEN"
            continue
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        
        res_str = f"{w}x{h}"
        resolutions[res_str] += 1
        durations.append(duration)
        fps_list.append(fps)
        
        vrec["width"] = w
        vrec["height"] = h
        vrec["resolution"] = res_str
        vrec["fps"] = round(fps, 2)
        vrec["frame_count"] = frame_count
        vrec["duration_s"] = round(duration, 2)
        
        # Sample 5 frames across video using direct seek
        sample_indices = np.linspace(0, max(0, frame_count - 1), 5, dtype=int)
        vid_lapvars = []
        vid_brightness = []
        read_success = 0
        for sidx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sidx))
            ret, frame = cap.read()
            if ret and frame is not None:
                read_success += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vid_lapvars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                vid_brightness.append(float(np.mean(gray)))
        cap.release()

        if read_success < len(sample_indices):
            vrec["decode_status"] = f"PARTIAL_READ_{read_success}_OF_5"
        else:
            vrec["decode_status"] = "SUCCESS"
            
        mean_lapvar = float(np.mean(vid_lapvars)) if vid_lapvars else 0.0
        mean_bright = float(np.mean(vid_brightness)) if vid_brightness else 0.0
        vrec["mean_laplacian_var"] = round(mean_lapvar, 2)
        vrec["mean_brightness"] = round(mean_bright, 2)
        
        sharpness_scores.append(mean_lapvar)
        brightness_scores.append(mean_bright)
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(video_records):
            print(f"  Processed {idx + 1}/{len(video_records)} videos...")

    print(f"Total videos audited: {len(video_records)}")
    print(f"Corrupted videos (cannot open): {len(corrupted_videos)}")
    print(f"Resolution breakdown: {dict(resolutions)}")
    print(f"Duration (s): min={np.min(durations):.2f}, mean={np.mean(durations):.2f}, max={np.max(durations):.2f}")
    print(f"FPS: min={np.min(fps_list):.2f}, mean={np.mean(fps_list):.2f}, max={np.max(fps_list):.2f}")
    print(f"Sharpness (Laplacian Var): min={np.min(sharpness_scores):.2f}, mean={np.mean(sharpness_scores):.2f}, max={np.max(sharpness_scores):.2f}")

    # -------------------------------------------------------------
    # 3. LANDMARK QUALITY AUDIT
    # -------------------------------------------------------------
    print("\n--- 3. Auditing Landmark Quality (862 npz archives) ---")
    landmark_archives = []
    class_lm_stats = defaultdict(lambda: {
        "archives": 0, "frames": 0, "slot0_present": 0, "slot1_present": 0,
        "both_present": 0, "zero_present": 0, "consec_slot0": 0, "consec_slot1": 0,
        "coord_out_of_bounds": 0, "timestamp_errors": 0
    })
    
    total_lm_archives = 0
    total_lm_frames = 0
    total_slot0_pres = 0
    total_slot1_pres = 0
    total_both_pres = 0
    total_zero_pres = 0
    total_consec_s0 = 0
    total_consec_s1 = 0
    coord_out_of_bounds_count = 0
    timestamp_error_count = 0
    non_finite_coords = 0

    for cf in class_folders:
        cf_dir = os.path.join(RAW_LANDMARKS_DIR, cf)
        if not os.path.exists(cf_dir):
            continue
        npz_files = sorted([f for f in os.listdir(cf_dir) if f.endswith(".npz")])
        for npz_file in npz_files:
            npz_path = os.path.join(cf_dir, npz_file)
            data = np.load(npz_path)
            
            # expected keys: landmarks, presence_mask, timestamps_ms, handedness
            lm = data["landmarks"]            # [T, 2, 21, 3]
            pres = data["presence_mask"]      # [T, 2] bool
            ts = data["timestamps_ms"]        # [T] int64
            
            T = len(ts)
            total_lm_archives += 1
            total_lm_frames += T
            
            s0_pres = int(np.sum(pres[:, 0]))
            s1_pres = int(np.sum(pres[:, 1]))
            both_pres = int(np.sum(pres[:, 0] & pres[:, 1]))
            zero_pres = int(np.sum((~pres[:, 0]) & (~pres[:, 1])))
            
            # Consecutive frames for velocity
            c_s0 = int(np.sum(pres[:-1, 0] & pres[1:, 0])) if T > 1 else 0
            c_s1 = int(np.sum(pres[:-1, 1] & pres[1:, 1])) if T > 1 else 0
            
            # Validity checks
            if not np.all(np.isfinite(lm)):
                non_finite_coords += 1
                
            # x, y normalized range checks for detected hands only
            oob = 0
            for s in [0, 1]:
                if np.any(pres[:, s]):
                    active_x = lm[pres[:, s], s, :, 0]
                    active_y = lm[pres[:, s], s, :, 1]
                    oob += int(np.sum((active_x < 0.0) | (active_x > 1.0) | (active_y < 0.0) | (active_y > 1.0)))
            if oob > 0:
                coord_out_of_bounds_count += int(oob)
                
            # Timestamp monotonicity
            ts_err = 0
            if T > 1:
                ts_diffs = np.diff(ts)
                if np.any(ts_diffs <= 0):
                    ts_err = int(np.sum(ts_diffs <= 0))
                    timestamp_error_count += ts_err
                    
            # Update totals
            total_slot0_pres += s0_pres
            total_slot1_pres += s1_pres
            total_both_pres += both_pres
            total_zero_pres += zero_pres
            total_consec_s0 += c_s0
            total_consec_s1 += c_s1
            
            # Per class update
            st = class_lm_stats[cf]
            st["archives"] += 1
            st["frames"] += T
            st["slot0_present"] += s0_pres
            st["slot1_present"] += s1_pres
            st["both_present"] += both_pres
            st["zero_present"] += zero_pres
            st["consec_slot0"] += c_s0
            st["consec_slot1"] += c_s1
            st["coord_out_of_bounds"] += int(oob)
            st["timestamp_errors"] += ts_err

    print(f"Total landmark archives: {total_lm_archives}")
    print(f"Total landmark frames: {total_lm_frames}")
    print(f"Left Hand (Slot 0) Presence: {total_slot0_pres / total_lm_frames * 100:.2f}%")
    print(f"Right Hand (Slot 1) Presence: {total_slot1_pres / total_lm_frames * 100:.2f}%")
    print(f"Both Hands Presence: {total_both_pres / total_lm_frames * 100:.2f}%")
    print(f"Zero Hands Presence: {total_zero_pres / total_lm_frames * 100:.2f}%")
    print(f"Non-finite coordinates: {non_finite_coords}")
    print(f"Normalized coord OOB instances: {coord_out_of_bounds_count}")
    print(f"Timestamp non-monotonic errors: {timestamp_error_count}")

    # -------------------------------------------------------------
    # 4. MODEL PERFORMANCE AUDIT
    # -------------------------------------------------------------
    print("\n--- 4. Auditing Model Performance (Train/Val/Test Splits & Evaluations) ---")
    with open(os.path.join(PROCESSED_DIR, "label_mapping.json")) as f:
        label_mapping = json.load(f)
    id_to_label = label_mapping["id_to_label"]
    label_to_id = label_mapping["label_to_id"]
    
    train_data = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    val_data = np.load(os.path.join(PROCESSED_DIR, "val.npz"))
    test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    
    train_y = train_data["y"]
    val_y = val_data["y"]
    test_y = test_data["y"]
    
    train_counts = Counter(train_y)
    val_counts = Counter(val_y)
    test_counts = Counter(test_y)
    
    with open(os.path.join(REPORTS_DIR, "evaluation_softmax_regression.json")) as f:
        eval_sr = json.load(f)
    with open(os.path.join(REPORTS_DIR, "evaluation_nearest_centroid.json")) as f:
        eval_nc = json.load(f)
        
    sr_per_class = eval_sr["per_class"]
    nc_per_class = eval_nc["per_class"]
    
    cm = np.array(eval_sr["confusion_matrix"])
    
    class_perf_records = []
    weak_classes = []
    
    for str_cid, cname in sorted(id_to_label.items(), key=lambda x: int(x[0])):
        cid = int(str_cid)
        tr_c = int(train_counts.get(cid, 0))
        va_c = int(val_counts.get(cid, 0))
        te_c = int(test_counts.get(cid, 0))
        total_c = tr_c + va_c + te_c
        
        sr_stats = sr_per_class.get(cname, {})
        nc_stats = nc_per_class.get(cname, {})
        
        prec = sr_stats.get("precision", 0.0)
        rec = sr_stats.get("recall", 0.0)
        f1 = sr_stats.get("f1_score", 0.0)
        
        # Identify incorrect predictions from confusion matrix
        # row cid = actual class, col pred_cid = predicted class
        row = cm[cid]
        false_negatives = []
        for pred_cid, count in enumerate(row):
            if pred_cid != cid and count > 0:
                false_negatives.append({
                    "predicted_id": pred_cid,
                    "predicted_class": id_to_label[str(pred_cid)],
                    "count": int(count)
                })
                
        # Column cid = predicted as cid, row actual_cid
        col = cm[:, cid]
        false_positives = []
        for actual_cid, count in enumerate(col):
            if actual_cid != cid and count > 0:
                false_positives.append({
                    "actual_id": actual_cid,
                    "actual_class": id_to_label[str(actual_cid)],
                    "count": int(count)
                })
                
        is_weak = (f1 < 0.60 or rec < 0.50)
        if is_weak:
            weak_classes.append({
                "class_id": cid,
                "class_name": cname,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "support": te_c,
                "false_negatives": false_negatives,
                "false_positives": false_positives
            })
            
        class_perf_records.append({
            "class_id": cid,
            "class_name": cname,
            "train_count": tr_c,
            "val_count": va_c,
            "test_count": te_c,
            "total_count": total_c,
            "sr_precision": prec,
            "sr_recall": rec,
            "sr_f1": f1,
            "nc_precision": nc_stats.get("precision", 0.0),
            "nc_recall": nc_stats.get("recall", 0.0),
            "nc_f1": nc_stats.get("f1_score", 0.0),
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "is_weak": is_weak
        })
        
    print(f"Overall Softmax Accuracy: {eval_sr['overall_accuracy'] * 100:.2f}%")
    print(f"Total weak classes (F1 < 0.60 or Recall < 0.50): {len(weak_classes)}")
    for wc in weak_classes:
        print(f"  Weak class: {wc['class_name']} (ID {wc['class_id']}) -> Recall: {wc['recall']:.2f}, F1: {wc['f1_score']:.2f}, Confusions: {[fn['predicted_class'] for fn in wc['false_negatives']]}")

    # -------------------------------------------------------------
    # 5. SIGNER AND DATASET DIVERSITY
    # -------------------------------------------------------------
    print("\n--- 5. Auditing Signer and Dataset Diversity ---")
    signer_summary = defaultdict(lambda: {
        "total_videos": 0, "mp4_count": 0, "mov_count": 0,
        "classes_covered": set(), "resolutions": Counter(),
        "mean_durations": []
    })
    
    for vrec in video_records:
        s = vrec["signer_group"]
        st = signer_summary[s]
        st["total_videos"] += 1
        if vrec["extension"] == ".mp4":
            st["mp4_count"] += 1
        else:
            st["mov_count"] += 1
        st["classes_covered"].add(vrec["class_name"])
        st["resolutions"][vrec.get("resolution", "unknown")] += 1
        st["mean_durations"].append(vrec.get("duration_s", 0.0))
        
    signer_report = {}
    for s, st in sorted(signer_summary.items(), key=lambda x: x[1]["total_videos"], reverse=True):
        signer_report[s] = {
            "total_videos": st["total_videos"],
            "mp4_count": st["mp4_count"],
            "mov_count": st["mov_count"],
            "class_coverage_count": len(st["classes_covered"]),
            "class_coverage_pct": round(len(st["classes_covered"]) / total_classes * 100, 2),
            "primary_resolutions": dict(st["resolutions"].most_common(2)),
            "avg_duration_s": round(float(np.mean(st["mean_durations"])), 2)
        }
        print(f"Signer: {s:25s} | Videos: {st['total_videos']:3d} (MP4: {st['mp4_count']:3d}, MOV: {st['mov_count']:3d}) | Classes: {len(st['classes_covered']):2d}/49 ({len(st['classes_covered'])/49*100:5.1f}%) | Res: {dict(st['resolutions'].most_common(1))}")

    # Assess signer-independent splitting feasibility
    # Can we leave one signer out? E.g., if we hold out Signer_0403, what classes have 0 samples left?
    # Let's compute class coverage intersection.
    all_class_names = set(class_folders)
    holdout_feasibility = {}
    for s, st in signer_summary.items():
        remaining_classes = set()
        for other_s, other_st in signer_summary.items():
            if other_s != s:
                remaining_classes.update(other_st["classes_covered"])
        missing_if_held_out = all_class_names - remaining_classes
        holdout_feasibility[s] = {
            "can_hold_out_cleanly": len(missing_if_held_out) == 0,
            "missing_classes_count": len(missing_if_held_out),
            "missing_classes": sorted(list(missing_if_held_out))
        }

    print("\nSigner Holdout Feasibility (Leave-One-Signer-Out):")
    for s, hf in holdout_feasibility.items():
        status = "FEASIBLE" if hf["can_hold_out_cleanly"] else f"INFEASIBLE (Missing {hf['missing_classes_count']} classes)"
        print(f"  {s:25s}: {status}")

    # -------------------------------------------------------------
    # 6. MISSING DATASET MATERIAL (THE 116 MOV FILES)
    # -------------------------------------------------------------
    print("\n--- 6. Auditing Missing Dataset Material (116 MOV Files) ---")
    mov_records = [v for v in video_records if v["extension"] == ".mov"]
    print(f"Total MOV files: {len(mov_records)}")
    
    mov_by_class = defaultdict(list)
    for m in mov_records:
        mov_by_class[m["class_name"]].append(m)
        
    print(f"Classes with MOV files: {len(mov_by_class)}")
    
    # Check decodability of all 116 MOV files
    mov_decodable = 0
    mov_failed = []
    mov_fps = []
    mov_resolutions = Counter()
    mov_durations = []
    
    for m in mov_records:
        cap = cv2.VideoCapture(m["filepath"])
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                mov_decodable += 1
                mov_fps.append(cap.get(cv2.CAP_PROP_FPS))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = fc / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                mov_resolutions[f"{w}x{h}"] += 1
                mov_durations.append(dur)
            else:
                mov_failed.append(m["filepath"])
        else:
            mov_failed.append(m["filepath"])
        cap.release()
        
    print(f"MOV Decodability: {mov_decodable}/{len(mov_records)} ({mov_decodable/len(mov_records)*100:.1f}%)")
    print(f"Failed MOV files: {len(mov_failed)}")
    print(f"MOV Resolutions: {dict(mov_resolutions)}")
    print(f"MOV Avg Duration: {np.mean(mov_durations):.2f}s, Avg FPS: {np.mean(mov_fps):.2f}")
    
    # Specifically inspect NachikethFriend1..3.mov
    friend_movs = [m for m in mov_records if m["class_name"] == "17.friend"]
    print("\nNachiketh Friend MOV Files:")
    for fm in friend_movs:
        print(f"  {fm['filename']} | Size: {fm['size_bytes']:,} bytes | Res: {fm.get('resolution')} | Dur: {fm.get('duration_s')}s | Status: {fm.get('decode_status')}")

    # -------------------------------------------------------------
    # 7. ASSEMBLE AUDIT JSON REPORT
    # -------------------------------------------------------------
    audit_data = {
        "audit_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "duration_seconds": round(time.time() - start_time, 2),
            "script": "ai-training/audit_phase8a.py",
            "python_version": sys.version
        },
        "dataset_inventory": {
            "raw_video_dir": RAW_VIDEO_DIR,
            "total_classes": total_classes,
            "missing_class_numbers": missing_numbers,
            "total_raw_files": total_raw_videos,
            "extension_breakdown": dict(ext_counter),
            "exact_duplicate_groups_count": len(exact_duplicate_groups),
            "intra_class_duplicates_count": len(intra_class_duplicates),
            "inter_class_duplicates_count": len(inter_class_duplicates),
            "inter_class_duplicates": inter_class_duplicates,
            "intra_class_duplicates_sample": intra_class_duplicates[:10]
        },
        "video_quality": {
            "total_videos_audited": len(video_records),
            "corrupted_videos_count": len(corrupted_videos),
            "resolution_distribution": dict(resolutions),
            "duration_stats_seconds": {
                "min": round(float(np.min(durations)), 2),
                "mean": round(float(np.mean(durations)), 2),
                "median": round(float(np.median(durations)), 2),
                "max": round(float(np.max(durations)), 2)
            },
            "fps_stats": {
                "min": round(float(np.min(fps_list)), 2),
                "mean": round(float(np.mean(fps_list)), 2),
                "max": round(float(np.max(fps_list)), 2)
            },
            "sharpness_laplacian_var": {
                "min": round(float(np.min(sharpness_scores)), 2),
                "mean": round(float(np.mean(sharpness_scores)), 2),
                "max": round(float(np.max(sharpness_scores)), 2)
            },
            "brightness_mean": {
                "min": round(float(np.min(brightness_scores)), 2),
                "mean": round(float(np.mean(brightness_scores)), 2),
                "max": round(float(np.max(brightness_scores)), 2)
            }
        },
        "landmark_quality": {
            "total_landmark_archives": total_lm_archives,
            "total_landmark_frames": total_lm_frames,
            "slot0_left_presence_rate": round(total_slot0_pres / total_lm_frames, 4),
            "slot1_right_presence_rate": round(total_slot1_pres / total_lm_frames, 4),
            "both_hands_presence_rate": round(total_both_pres / total_lm_frames, 4),
            "zero_hands_presence_rate": round(total_zero_pres / total_lm_frames, 4),
            "consecutive_slot0_velocity_mask_rate": round(total_consec_s0 / total_lm_frames, 4),
            "consecutive_slot1_velocity_mask_rate": round(total_consec_s1 / total_lm_frames, 4),
            "non_finite_coordinates": non_finite_coords,
            "coordinate_out_of_bounds_count": coord_out_of_bounds_count,
            "timestamp_non_monotonic_errors": timestamp_error_count
        },
        "model_performance": {
            "overall_accuracy": eval_sr["overall_accuracy"],
            "macro_f1": eval_sr["macro_avg"]["f1_score"],
            "weighted_f1": eval_sr["weighted_avg"]["f1_score"],
            "weak_classes_count": len(weak_classes),
            "weak_classes": weak_classes,
            "per_class_performance": class_perf_records
        },
        "signer_diversity": {
            "signer_report": signer_report,
            "holdout_feasibility": holdout_feasibility
        },
        "missing_dataset_material": {
            "total_mov_files": len(mov_records),
            "classes_with_mov_files": len(mov_by_class),
            "mov_decodable_count": mov_decodable,
            "mov_failed_count": len(mov_failed),
            "mov_resolutions": dict(mov_resolutions),
            "nachiketh_friend_mov_files": [
                {
                    "filename": fm["filename"],
                    "size_bytes": fm["size_bytes"],
                    "resolution": fm.get("resolution"),
                    "fps": fm.get("fps"),
                    "duration_s": fm.get("duration_s")
                }
                for fm in friend_movs
            ]
        },
        "per_class_summary": {}
    }
    
    # Build complete per-class summary combining inventory, landmarks, and model metrics
    for cf in class_folders:
        cid = label_to_id.get(cf, -1)
        lm_st = class_lm_stats.get(cf, {})
        perf_rec = next((p for p in class_perf_records if p["class_name"] == cf), {})
        inv_st = class_inventory.get(cf, {})
        
        frames_c = lm_st.get("frames", 0)
        s0_rate = round(lm_st.get("slot0_present", 0) / frames_c, 4) if frames_c > 0 else 0.0
        s1_rate = round(lm_st.get("slot1_present", 0) / frames_c, 4) if frames_c > 0 else 0.0
        both_rate = round(lm_st.get("both_present", 0) / frames_c, 4) if frames_c > 0 else 0.0
        zero_rate = round(lm_st.get("zero_present", 0) / frames_c, 4) if frames_c > 0 else 0.0
        
        audit_data["per_class_summary"][cf] = {
            "class_id": cid,
            "class_name": cf,
            "raw_total_files": inv_st.get("total_files", 0),
            "raw_mp4_files": inv_st.get("mp4_count", 0),
            "raw_mov_files": inv_st.get("mov_count", 0),
            "extracted_archives": lm_st.get("archives", 0),
            "total_frames": frames_c,
            "slot0_presence_rate": s0_rate,
            "slot1_presence_rate": s1_rate,
            "both_hands_presence_rate": both_rate,
            "zero_hands_presence_rate": zero_rate,
            "train_samples": perf_rec.get("train_count", 0),
            "val_samples": perf_rec.get("val_count", 0),
            "test_samples": perf_rec.get("test_count", 0),
            "test_precision": perf_rec.get("sr_precision", 0.0),
            "test_recall": perf_rec.get("sr_recall", 0.0),
            "test_f1": perf_rec.get("sr_f1", 0.0),
            "is_weak": perf_rec.get("is_weak", False),
            "confusions": perf_rec.get("false_negatives", [])
        }

    output_json_path = os.path.join(REPORTS_DIR, "phase8a_complete_dataset_audit.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nSaved complete audit JSON to: {output_json_path}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    audit_dataset()

