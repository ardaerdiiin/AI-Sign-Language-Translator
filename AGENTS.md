# AGENTS.md

## Must-follow constraints

- Face landmarks are **excluded** from the model pipeline. Feature vector is **194 engineered features**: upper-body pose (8×3=24) + wrist-centered hand coords (20×3×2=120) + joint angles (15×2=30) + fingertip distances (10×2=20). Do not add face data or raw 258-dim vectors back.
- The active pipeline is strictly: `process_videos.py` → `train_model.py` → `convert_to_tflite.py` → `main.py`. All unused legacy scripts (e.g. `record_data.py`, `_legacy/`) and old debuggers have been purged from the codebase.
- `extract_keypoints()` and `mediapipe_detection()` live in **`src/keypoints_utils.py`** (Single Source of Truth). Both `process_videos.py` and `src/inference.py` import from there. Any keypoint change goes in `keypoints_utils.py` only.
- Model input shape is `(16, 194)` — 16 frames, 194 engineered features. Architecture: **Transformer Encoder** (d_model=512, nhead=4, num_layers=2) inspired by TSLFormer (arXiv:2505.07890v4). Changing feature count requires retraining.
- `action.h5` is the trained model weights file at repo root. Never delete or overwrite without retraining.
- **Label mapping is in `labels.json`** at repo root. Labels are NOT auto-discovered from filesystem. To add a new word: add videos to `raw_videos/WORD/`, run `process_videos.py` (auto-updates `labels.json`), then retrain.

## Repo-specific conventions

- All paths use `Path(__file__).resolve().parent` (absolute) — the app can be run from **any directory**.
- Data pipeline order: place videos in `raw_videos/WORD_NAME/` → `process_videos.py` → `train_model.py` → run `main.py`.
- `raw_videos/` structure: one subdirectory per word, containing `.mp4`/`.avi`/`.mov`/`.mkv` files.
- `data/` structure: `data/WORD/SEQUENCE_NUM/FRAME.npy` — each `.npy` file is a 194-dim feature vector.
- UI is PyQt5. Camera uses a **2-stage Producer-Consumer threading model**: `CameraReader` (I/O only, Queue maxsize=1) → `VideoThread` (MediaPipe & Prediction). Avatar video playback uses `QTimer` (non-blocking).

## Important locations

- `src/keypoints_utils.py` — keypoint extraction & mediapipe detection (Single Source of Truth)
- `src/model.py` — Transformer Encoder model architecture with sinusoidal positional encoding (change here requires retrain + weight reset)
- `src/inference.py` — real-time prediction logic, stability filtering, reads `labels.json`
- `src/avatar_module.py` — text-to-sign video lookup from `raw_videos/`
- `process_videos.py` — offline video-to-keypoint pipeline, auto-updates `labels.json`
- `train_model.py` — model training with train/val split, reads `labels.json`, saves `.h5`
- `convert_to_tflite.py` — Converts trained `action.h5` to quantized `action.tflite` for ~7x faster inference (3ms vs 20ms).
- `labels.json` — deterministic label-to-index mapping

## Known gotchas

- `train_model.py` uses 85/15 train/val split. `val_categorical_accuracy` is the monitored metric.
- Prediction threshold is `self.threshold = 0.40` and margin threshold is `self.margin_threshold = 0.05` in `SignLanguagePredictor` — single source, no magic numbers. Margin filter prevents overconfident wrong predictions on visually similar signs.
- Feature engineering uses `EPS = 1e-7` in `keypoints_utils.py` to guard against ZeroDivisionError when hands are undetected.
- `extract_raw_landmarks()` is NOT affected by feature engineering — it provides raw coords for Unity avatar.
- Adding a new word requires both `raw_videos/WORD/` data AND running `process_videos.py` to update `labels.json`.
