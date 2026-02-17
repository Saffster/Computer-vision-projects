import time
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# FER+ label order
EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def preprocess_face_bgr(face_bgr: np.ndarray) -> np.ndarray:
    # FER+ expects 64x64 grayscale, shape (1,1,64,64)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32).reshape(1, 1, 64, 64)
    return x

def detection_score(det) -> float:
    # MediaPipe Tasks detection score is typically in det.categories[0].score
    try:
        if det.categories and det.categories[0].score is not None:
            return float(det.categories[0].score)
    except Exception:
        pass
    return 0.0

def main():
    # Files you must have:
    # 1) models/emotion-ferplus-8.onnx
    # 2) models/blaze_face_short_range.tflite
    onnx_path = "models/emotion-ferplus-8.onnx"
    face_model_path = "models/blaze_face_short_range.tflite"

    # ONNX Runtime session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # MediaPipe Tasks FaceDetector (VIDEO mode)
    base_options = mp_python.BaseOptions(model_asset_path=face_model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.6,
    )
    detector = vision.FaceDetector.create_from_options(options)

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    smooth_window = 8
    prob_hist = []

    prev_t = time.time()
    fps = 0.0

    # Use a monotonically increasing timestamp for VIDEO mode
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result and result.detections:
            # pick the most confident detection
            det = max(result.detections, key=detection_score)
            bbox = det.bounding_box  # pixel coords

            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)

            # padding helps expression model
            pad = int(0.20 * max(bw, bh))
            x1p = clamp(x1 - pad, 0, w - 1)
            y1p = clamp(y1 - pad, 0, h - 1)
            x2p = clamp(x1 + bw + pad, 0, w - 1)
            y2p = clamp(y1 + bh + pad, 0, h - 1)

            face = frame[y1p:y2p, x1p:x2p]
            if face.size > 0:
                x_in = preprocess_face_bgr(face)
                scores = sess.run(None, {input_name: x_in})[0].astype(np.float32).reshape(-1)
                probs = softmax(scores)

                prob_hist.append(probs)
                if len(prob_hist) > smooth_window:
                    prob_hist.pop(0)

                probs_s = np.mean(np.stack(prob_hist, axis=0), axis=0)
                idx = int(np.argmax(probs_s))
                label = EMOTIONS[idx]
                conf = float(probs_s[idx])

                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1p, max(0, y1p - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # FPS display
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Real-time Emotion (FER+)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
