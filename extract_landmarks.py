import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import time

# Setup MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

dataset_path = r"D:\dataset\asl_alphabet_train\asl_alphabet_train"
output_csv = "asl_landmarks_full.csv"

folders = sorted(os.listdir(dataset_path))
print(f"Found {len(folders)} folders: {folders}")
print("Processing ALL 3000 images per letter...")
print("Estimated time: 15-20 minutes. Go grab a coffee! ☕\n")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")
    writer.writerow(header)

    total_saved = 0
    start_time = time.time()

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        label = folder.upper()
        all_images = sorted(os.listdir(folder_path))
        folder_saved = 0
        folder_start = time.time()

        print(f"Processing: {label} ({len(all_images)} images)...")

        for img_file in all_images:
            img_path = os.path.join(folder_path, img_file)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    row = []
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label)
                    writer.writerow(row)
                    folder_saved += 1
                    total_saved += 1

            except Exception:
                continue

        folder_time = time.time() - folder_start
        elapsed = time.time() - start_time
        print(f"✅ {label}: {folder_saved}/{len(all_images)} saved | "
              f"Folder time: {folder_time:.1f}s | "
              f"Total: {total_saved} | "
              f"Elapsed: {elapsed:.1f}s")

elapsed = time.time() - start_time
print(f"\n🎉 All done!")
print(f"📊 Total samples: {total_saved}")
print(f"⏱️ Total time: {elapsed:.1f} seconds")
print(f"📁 Saved to: {output_csv}")