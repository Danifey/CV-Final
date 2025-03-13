import cv2
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern  # For LBP feature extraction

# Configuration Constants
CAT_FOLDERS = [
    "ingest footage/freeway",
    "ingest footage/rengar",
    "ingest footage/Yumi"
]
HAAR_CASCADE_PATH = 'haarcascade_frontalcatface.xml'
GRID_SIZE = (4, 4)  # Split face into 4x4 grid
FACE_SIZE = (100, 100)  # Standard size for face resizing


def extract_features(frame):
    """
    Extract distinguishing features from a frame containing a cat face
    Returns feature vector or None if no face detected
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cat faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Take largest face (assuming single cat per frame)
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = frame[y:y + h, x:x + w]

    # Resize face to standard size for consistent feature extraction
    face_roi = cv2.resize(face_roi, FACE_SIZE)

    # --- Color Features (HSV Space) ---
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    cell_height = FACE_SIZE[0] // GRID_SIZE[0]
    cell_width = FACE_SIZE[1] // GRID_SIZE[1]

    hsv_features = []
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            # Extract grid cell
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            cell = hsv[y_start:y_end, x_start:x_end]
            # Calculate mean HSV values for the cell
            hsv_features.extend(np.mean(cell, axis=(0, 1)))

    # --- Texture Features (LBP) ---
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(
        gray_face,
        P=8,  # 8 surrounding points
        R=1,  # Radius of 1 pixel
        method='uniform'
    )
    # Create histogram of LBP patterns (16 bins)
    lbp_hist, _ = np.histogram(
        lbp,
        bins=16,
        range=(0, 16)
    )

    # Combine all features into single array
    return np.concatenate([hsv_features, lbp_hist])


# Initialize face detector
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    raise FileNotFoundError("Could not load Haar Cascade file. Check the path.")

# Initialize DataFrame to store features
feature_data = []

# Process videos from each cat's folder
for cat_folder in CAT_FOLDERS:
    # Get label from folder name (assuming last part is cat name)
    label = os.path.basename(os.path.normpath(cat_folder))

    # Process each video in the folder
    for video_file in os.listdir(cat_folder):
        video_path = os.path.join(cat_folder, video_file)
        print(f"Processing {video_path}...")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract features from frame
            features = extract_features(frame)

            if features is not None:
                # Add features with label to dataset
                feature_data.append([*features, label])
                frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file}")

# Determine the number of HSV and LBP features dynamically
example_features = extract_features(np.zeros((100, 100, 3), dtype=np.uint8))  # Dummy frame
num_hsv_features = GRID_SIZE[0] * GRID_SIZE[1] * 3  # 3 channels (H, S, V)
num_lbp_features = 16  # 16 bins for LBP histogram

# Create DataFrame and save to CSV
columns = (
    [f"hsv_{i}" for i in range(num_hsv_features)] +
    [f"lbp_{i}" for i in range(num_lbp_features)] +
    ["label"]
)
df = pd.DataFrame(feature_data, columns=columns)
df.to_csv("cat_features.csv", index=False)
print("Dataset saved to cat_features.csv")

# Train Random Forest classifier
X = df.drop('label', axis=1).values
y = df['label'].values

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Initialize and train classifier
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate performance
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

# Save training and validation accuracy to a .txt file
with open("training_validation_accuracy.txt", "w") as f:
    f.write(f"Training Accuracy: {train_acc:.2f}\n")
    f.write(f"Validation Accuracy: {test_acc:.2f}\n")

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {test_acc:.2f}")
print("Training and validation accuracy saved to training_validation_accuracy.txt")

# Save model for later use
import joblib

joblib.dump(clf, 'cat_classifier.pkl')
print("Model saved as cat_classifier.pkl")