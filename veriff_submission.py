
# =============================================================
 # Veriff - Take Home Assignment - V4 (with Yolo V10)
 # Classifies each video as 'Single Person' or 'Multiple People'
# =============================================================


import os
import csv
import cv2
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import sys
import logging

class Tee:
    """Mirrors stdout to both terminal and a log file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._terminal = sys.stdout
        self._log = open(filepath, "w", buffering=1)  # line-buffered

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

# ===========================================================================================
#  This is my config area
#  Change these values to adjust how the system behaves, input-output locations and also YOLO settings 
# =====================================================================================================

VIDEO_FOLDER         = "videos"                    # Folder with the .mp4 files by Veriff 
OUTPUT_CSV           = "output/results.csv"        # Where Results are stored
EVALUATION_CSV       = "output/evaluation.csv"     # Where accuracy comparison will be saved
ANNOTATED_DIR        = "output/annotated_frames"   # Store Annoted Frames for checking and understanding later where things are failing
LABELS_FILE          = "labels.txt"                # The file by Veriff with True Labels to check later on

CONFIDENCE           = 0.6    # Minimum confidence for YOLO to count a detection (0.0 to 1.0)

PERSON_CLASS_ID      = 0      # In YOLO's COCO dataset,  0 is 'person' , I will not change this

SAMPLE_FPS           = 1      # How many frames to analyze per second of video, I have kept this low to lower processing time

MIN_FRAMES_FOR_MULTI = 2      # How many frames must show ≥2 persons  to classify the video as "Multiple People"
                              # Higher valur to make it more strict and reduce False Positives 

SAVE_ANNOTATED       = True   # True  = To save frames with bounding boxes drawn (for checking later)

MODEL_WEIGHTS        = "yolov10m.pt"   # YOLOv10 model to use, and also which one. n is for Nano, m is medium, l is large. I played around this as well 


# ======================================================================================================================
# FunctionP 1: Model Loading -> Loading the YOLOv10 model and running on all videos 
# Another Context for the step 1 - I use a M4 Mac Mini, hence I am using Torch to detect and use it otherwise use CPU 
# =======================================================================================================================

def load_model(weights: str = MODEL_WEIGHTS) -> YOLO:
    """
    Loading the pretrained YOLOv10 model
    Tourch checks if M4 GPU is available (in my case), else CPU can be used.
    """
    print(f"\n[INFO] Loading YOLOv10 model: {weights}")

    model = YOLO(weights)  # Downloads the model automatically if not already present

    # Detect my available hardware and uses the best one
    # MPS = Apple's M4 — better than CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)
    print(f"[INFO] Running on device: {device}")

    return model


# =============================================================
#  Function 2: DETECT PEOPLE IN ONE FRAME
#  Runs YOLO on a single image/frame and returns person count
# =============================================================

def detect_people(model: YOLO, frame) -> tuple:
    """
    This function receives a YOLO Model and a frame and it returns a tuple
    Input: Model and Frame 
    Output:
        person_count: number of people found in the frame
        boxes: list of bounding boxes as (x1, y1, x2, y2)
    """
    results = model.predict(
        source=frame,
        classes=[PERSON_CLASS_ID],   # person only
        conf=CONFIDENCE,             # skip low-confidence hits, check 
        verbose=False                # I am keeping it False for now, low amount of log 
    )

    boxes = []

    for result in results:
        for box in result.boxes:
            # safety check in case anything else slips through
            if int(box.cls[0]) == PERSON_CLASS_ID: # double checking if the person is a person 
                # YOLO gives box corners as x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    return len(boxes), boxes


# =============================================================================
#  Function 3: Drawing boundary boxes on a frame
#  this wokrs only when SAVE_ANNOTATED = True, this setting is also on the top
# =============================================================================

def annotate_frame(frame, boxes: list, frame_idx: int, person_count: int):
    """
    Drawing boxes and labels on the frame - This is was useful when I was trying to undersatnd what is working and what is not. 

    Returns:
        A copy of the frame with the summary and boxess
    """
    annotated = frame.copy()  # also keeping the original frame 

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # person box
        cv2.putText( # to wirte the label "Person" on the boxes
            annotated, "Person",  
            (x1, y1 - 8),  # put label just above the box
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    # this is very useful for debugging while reviewing what is working and what is not
    label = f"Frame {frame_idx} | Persons: {person_count}"
    cv2.putText(
        annotated, label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 0, 255), 2  # red stands out better here
    )

    return annotated

# =============================================================
#  Function 4: Process one video 
#  Opens the video, samples frames, runs detection, returns stats
# =============================================================
def process_video(model: YOLO, video_path: str) -> dict:
    """
    Process one video and return the main  stats.
    Here is a flow of this function:
    --> (1) : Extract File name and filename without extension (this is good for printable logs and outputs) 
    --> (2) : It opens the file with openCV, and exits early if the file can't be opend
    --> (3) : Reading of stats like FPS, total frame count, duration
    --> (4) : Computes frame_interval, I have set the Sample FPS = 1 on the top of the code 
    --> (5) : Initlize counters to get -> max people, frames with multiple people, how many frames were processed (simple step)
    --> (6) : Lopps through the video frame by frame 
    --> (7) : For each frame it calls our 2nd function detect_people, but runs only when frame number matches sampling interval
    --> (8) : It updates the running counters, and if that frame has 2 or more people, it optionally saves an annotated version for debugging
    --> (9) : After the loop ends, it releases the video file
    --> (10): It applies the classification rule and returns a dictionary with the final stats
    """
    #(1) - Check the description above 
    video_name = Path(video_path).name   # full file name, like "veriff1.mp4"
    video_stem = Path(video_path).stem   # file name without extension

    print(f"\n{'=' * 55}")
    print(f"[PROCESSING] {video_name}")
    print(f"{'=' * 55}")

    #(2) - Check the description above - open file with openCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return None

    #(3) - Check the description above - basic video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0

    print(f"  FPS          : {fps:.2f}")
    print(f"  Total Frames : {total_frames}")
    print(f"  Duration     : {duration_sec} seconds")

    # (4) - check the description above - Computes frame_interval instead of looking at all frames, SAMPLE_FPS defined at the top
    frame_interval = max(1, int(fps / SAMPLE_FPS))
    print(f"  Sampling every {frame_interval} frame(s) ({SAMPLE_FPS} frame/sec)\n")

    if SAVE_ANNOTATED:
        video_annotate_dir = os.path.join(ANNOTATED_DIR, video_stem)
        os.makedirs(video_annotate_dir, exist_ok=True)

    # (5) - check the description above, initialize counters
    max_person_count = 0
    frames_with_multiple = 0
    frames_processed = 0
    current_frame_idx = 0

    # (6) - check the description above, looping through frame by frame 
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        #(7) - Check the description above - calling the detect_people function when sampling rate match the frame number 
        if current_frame_idx % frame_interval == 0:
            person_count, boxes = detect_people(model, frame)
            frames_processed += 1

            if person_count > max_person_count:
                max_person_count = person_count

            if person_count >= 2:
                frames_with_multiple += 1

                #(8) - Check the description above - save the frame only when multiple people are present
                if SAVE_ANNOTATED:
                    annotated = annotate_frame(frame, boxes, current_frame_idx, person_count)
                    save_path = os.path.join(
                        video_annotate_dir,
                        f"frame_{current_frame_idx:06d}_persons_{person_count}.jpg"
                    )
                    cv2.imwrite(save_path, annotated)

            print(
                f"  [Frame {current_frame_idx:5d}] "
                f"Sample #{frames_processed:3d} | "
                f"Persons detected: {person_count}"
            )

        current_frame_idx += 1

    cap.release()

    #(10) - check the description above, mark as multiple people only if it happens often enough
    if frames_with_multiple >= MIN_FRAMES_FOR_MULTI:
        classification = "Multiple People"
    else:
        classification = "Single Person"

    print(f"\n  ── RESULT ──────────────────────────────")
    print(f"  Frames processed         : {frames_processed}")
    print(f"  Max persons in one frame : {max_person_count}")
    print(f"  Frames with ≥2 persons   : {frames_with_multiple}")
    print(f"  Classification           : {classification}")

    return {
        "video_name": video_name,
        "duration_sec": duration_sec,
        "frames_processed": frames_processed,
        "max_person_count": max_person_count,
        "frames_with_multiple_people": frames_with_multiple,
        "classification": classification,
    }

# =============================================================
#  Function 5: Save Results into a CSV 
#  Writes one row per video into results.csv
# =============================================================

def save_results(results: list, output_path: str):
    """
    Save the video-level results to a CSV file.
    """
    # make sure the output folder exists before writing the file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # keeping the CSV columns in a fixed order
    fieldnames = [
        "video_name",
        "duration_sec",
        "frames_processed",
        "max_person_count",
        "frames_with_multiple_people",
        "classification",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()      # column names
        writer.writerows(results) # one row per video

    print(f"\n[SAVED] Detection results → {output_path}")

# ===================================================================================
#  Function 6: Evalutate my detection with what Veriff has sent in the file labels.txt
# ===================================================================================

def evaluate_results(results_csv: str, labels_file: str):
    """
    Compare model predictions against the ground-truth labels.
    Print the summary and save the full evaluation table.
    """
    # skip evaluation if the labels file is missing
    if not os.path.exists(labels_file):
        print(f"\n[WARNING] Labels file not found: {labels_file} — skipping evaluation.")
        return

    # Load model's results
    results_df = pd.read_csv(results_csv)

    # convert text labels into 0/1 so they match the labels file
    results_df["predicted"] = results_df["classification"].apply(
        lambda x: 1 if x == "Multiple People" else 0
    )

    # Loading veriff's labels.txt
    labels_df = pd.read_csv(labels_file, sep="\t")  # Tab-separated
    labels_df.columns = ["video_stem", "actual"]    # Rename columns for clarity
    labels_df["video_stem"] = labels_df["video_stem"].str.strip()

    # add .mp4 so both files use the same video extension
    labels_df["video_name"] = labels_df["video_stem"] + ".mp4"

    # join predictions with actual labels using video_name
    merged = results_df.merge(labels_df[["video_name", "actual"]], on="video_name")

    # mark which predictions were correct
    merged["correct"] = merged["predicted"] == merged["actual"]

    # print evaluation table 
    print("\n" + "="*55)
    print(" EVALUATION RESULTS")
    print("="*55)
    print(merged[["video_name", "actual", "predicted", "correct"]].to_string(index=False))

    # accuracy summary
    accuracy = merged["correct"].mean() * 100
    total    = len(merged)
    correct  = merged["correct"].sum()
    print(f"\n Accuracy : {accuracy:.1f}%  ({correct}/{total} videos correct)")

    # ── Show misclassified videos ──────────────────────────
    wrong = merged[~merged["correct"]]
    if not wrong.empty:
        print(f"\n Misclassified ({len(wrong)} video(s)):")
        print(wrong[["video_name", "actual", "predicted"]].to_string(index=False))
    else:
        print("\n All videos classified correctly!")

    # ── Save full evaluation to CSV ────────────────────────
    merged.to_csv(EVALUATION_CSV, index=False)
    print(f"\n[SAVED] Evaluation details → {EVALUATION_CSV}")


# =============================================================
#  MAIN — This ties everything together
# =============================================================

def main():
    # ── Logging setup ──────────────────────────────────────────
    LOG_FILE = "output/run.log"
    tee = Tee(LOG_FILE)
    sys.stdout = tee                          # all print() → terminal + file
    # ──────────────────────────────────────────────────────────

    # ── Check that videos folder exists ───────────────────
    if not os.path.isdir(VIDEO_FOLDER):
        print(f"[ERROR] Videos folder not found: '{VIDEO_FOLDER}'")
        print("        Make sure you created the 'videos' folder inside veriff/")
        return

    # ── Find all .mp4 files in the videos folder ──────────
    video_files = sorted(Path(VIDEO_FOLDER).glob("*.mp4"))

    if not video_files:
        print(f"[ERROR] No .mp4 files found in '{VIDEO_FOLDER}' folder.")
        return

    print(f"[INFO] Found {len(video_files)} video(s) to process.")
    for v in video_files:
        print(f"       → {v.name}")

    # ── Load the YOLO model once (reused for all videos) ──
    model = load_model()

    # ── Process each video one by one ─────────────────────
    all_results = []
    for video_path in video_files:
        result = process_video(model, str(video_path))
        if result:
            all_results.append(result)

    # ── Save detection results to CSV ─────────────────────
    if all_results:
        save_results(all_results, OUTPUT_CSV)

    # ── Evaluate against your ground truth labels ─────────
    evaluate_results(OUTPUT_CSV, LABELS_FILE)

    print("\n[DONE] All videos processed. Check the output/ folder for results.")
    tee.close()                               # flush and close log file
    sys.stdout = tee._terminal                # restore normal stdout


# ── Entry point ───────────────────────────────────────────────
# This ensures main() only runs when you execute this file directly
# and not when it's imported as a module
if __name__ == "__main__":
    main()
