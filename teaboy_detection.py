import os
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# ============================================================
# Paths
# ============================================================
VIDEO_PATH = "/kaggle/input/datasets/arkasmitsengupta12/test-boy-ds1/teaboy_video2.mp4"
PERSON_MODEL_PATH = "yolo26n.pt"
CLASSIFIER_PATH = "/kaggle/working/classifier_runs/teaboy_classifier_resnet18_best.pt"

OUTPUT_DIR = "/kaggle/working"
RAW_OUTPUT = os.path.join(
    OUTPUT_DIR, "teaboy_video2_combined_output.mp4"
)
COMPRESSED_OUTPUT = os.path.join(
    OUTPUT_DIR, "teaboy_video2_combined_output_compressed.mp4"
)

# ============================================================
# Configuration
# ============================================================
MAX_SECONDS = None       # None = process the full video
PERSON_CONF = 0.4        # Minimum YOLO person confidence
IMG_SIZE = 224            # ResNet input size

CLASS_TO_IDX = {
    "not_teaboy": 0,
    "teaboy": 1,
}
TEABOY_IDX = CLASS_TO_IDX["teaboy"]

# ============================================================
# Vest color check
# HSV ranges obtained from vest-color calibration
# ============================================================
TORSO_TOP_FRAC = 0.15
TORSO_BOTTOM_FRAC = 0.55

LOWER_A = np.array([160, 60, 35])
UPPER_A = np.array([179, 150, 90])

LOWER_B = np.array([0, 60, 35])
UPPER_B = np.array([8, 150, 90])

# Minimum fraction of torso pixels matching the vest color
VEST_RATIO_THRESHOLD = 0.06

# ============================================================
# Device
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# ============================================================
# Load YOLO person detector
# ============================================================
person_model = YOLO(PERSON_MODEL_PATH)

# ============================================================
# Load ResNet-18 Teaboy Classifier
# ============================================================
classifier = models.resnet18(weights=None)

# Replace ImageNet's original classifier with 2-class classifier
classifier.fc = nn.Linear(
    classifier.fc.in_features,
    2
)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=device
    )
)

classifier.to(device)
classifier.eval()

# ============================================================
# Image preprocessing for ResNet-18
# ============================================================
classify_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def classify_crop(crop_bgr):
    """
    Classify a detected person's crop using ResNet-18.

    Returns:
        is_teaboy: True/False
        conf: probability assigned to the teaboy class
    """
    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    tensor = classify_tf(crop_rgb)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    is_teaboy = (
        probs.argmax().item() == TEABOY_IDX
    )

    conf = probs[TEABOY_IDX].item()

    return is_teaboy, conf


def vest_pixel_ratio(crop_bgr):
    """
    Calculate the fraction of torso pixels that match
    the calibrated vest color in HSV space.
    """
    h, w = crop_bgr.shape[:2]

    y1 = int(h * TORSO_TOP_FRAC)
    y2 = int(h * TORSO_BOTTOM_FRAC)

    torso = crop_bgr[y1:y2, :]

    if torso.size == 0:
        return 0.0

    hsv = cv2.cvtColor(
        torso,
        cv2.COLOR_BGR2HSV
    )

    mask_a = cv2.inRange(
        hsv,
        LOWER_A,
        UPPER_A
    )

    mask_b = cv2.inRange(
        hsv,
        LOWER_B,
        UPPER_B
    )

    mask = cv2.bitwise_or(
        mask_a,
        mask_b
    )

    ratio = (
        mask.sum()
        / 255
        / (torso.shape[0] * torso.shape[1])
    )

    return ratio


def main():
    # ========================================================
    # Open input video
    # ========================================================
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise SystemExit(
            f"ERROR: cannot open {VIDEO_PATH}"
        )

    w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    max_frames = (
        int(fps * MAX_SECONDS)
        if MAX_SECONDS
        else total_frames
    )

    # ========================================================
    # Output video writer
    # ========================================================
    writer = cv2.VideoWriter(
        RAW_OUTPUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # ========================================================
    # Counters
    # ========================================================
    frame_idx = 0
    teaboy_frame_count = 0
    total_persons_seen = 0

    classifier_said_yes_but_vest_said_no = 0

    # ========================================================
    # Process video frame by frame
    # ========================================================
    while frame_idx < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # YOLO detects only class 0 = person
        results = person_model(
            frame,
            verbose=False,
            classes=[0]
        )

        frame_had_teaboy = False

        for box in results[0].boxes:
            conf = float(box.conf[0])

            if conf < PERSON_CONF:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = frame[
                max(0, y1):y2,
                max(0, x1):x2
            ]

            if crop.size == 0:
                continue

            total_persons_seen += 1

            # ------------------------------------------------
            # Stage 1: ResNet-18 classification
            # ------------------------------------------------
            is_teaboy_clf, clf_conf = classify_crop(
                crop
            )

            # ------------------------------------------------
            # Stage 2: HSV vest-color verification
            # ------------------------------------------------
            vest_ratio = vest_pixel_ratio(
                crop
            )

            vest_says_yes = (
                vest_ratio >= VEST_RATIO_THRESHOLD
            )

            # Track cases where classifier says Teaboy
            # but vest-color check rejects it
            if is_teaboy_clf and not vest_says_yes:
                classifier_said_yes_but_vest_said_no += 1

            # ------------------------------------------------
            # Final decision:
            # BOTH classifier and vest check must agree
            # ------------------------------------------------
            is_teaboy_final = (
                is_teaboy_clf
                and vest_says_yes
            )

            if is_teaboy_final:
                frame_had_teaboy = True

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Teaboy clf{clf_conf:.2f} "
                    f"vest{vest_ratio:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # No box is drawn for rejected detections

        if frame_had_teaboy:
            teaboy_frame_count += 1

        # ----------------------------------------------------
        # Display overall frame status
        # ----------------------------------------------------
        cv2.rectangle(
            frame,
            (0, 0),
            (280, 50),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Teaboy present: "
            f"{'YES' if frame_had_teaboy else 'no'}",
            (10, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        writer.write(frame)
        frame_idx += 1

    # ========================================================
    # Release video resources
    # ========================================================
    cap.release()
    writer.release()

    # ========================================================
    # Print processing statistics
    # ========================================================
    print(
        f"\nProcessed {frame_idx} frames "
        f"(~{frame_idx / fps:.1f}s)"
    )

    print(
        f"Total person crops evaluated: "
        f"{total_persons_seen}"
    )

    print(
        f"Frames with a confirmed teaboy "
        f"(both signals agree): "
        f"{teaboy_frame_count}"
    )

    print(
        "Cases where classifier said teaboy "
        "but vest-color disagreed (blocked): "
        f"{classifier_said_yes_but_vest_said_no}"
    )

    print(
        f"Raw output saved to: {RAW_OUTPUT}"
    )

    # ========================================================
    # Compress output video using FFmpeg
    # ========================================================
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        RAW_OUTPUT,
        "-vcodec",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "medium",
        "-acodec",
        "aac",
        COMPRESSED_OUTPUT,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(
            "ffmpeg compression failed:\n"
            f"{result.stderr[-1000:]}"
        )
    else:
        before = (
            os.path.getsize(RAW_OUTPUT)
            / (1024 * 1024)
        )

        after = (
            os.path.getsize(COMPRESSED_OUTPUT)
            / (1024 * 1024)
        )

        print(
            f"Compressed: {COMPRESSED_OUTPUT} "
            f"({before:.1f}MB -> {after:.1f}MB)"
        )

        print(
            f"\nDownload from Kaggle output pane: "
            f"{COMPRESSED_OUTPUT}"
        )


if __name__ == "__main__":
    main()
