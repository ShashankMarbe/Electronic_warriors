import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if success:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

# Function to load and preprocess the extracted frames
def load_data(image_folder):
    images = []
    frame_paths = []
    
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))  # Resize to the input shape of the model
        images.append(img)
        frame_paths.append(img_path)
    
    images = np.array(images) / 255.0  # Normalize the images
    return images, frame_paths

# Function to classify frames and give response
def classify_frames(model, images, frame_paths):
    predictions = model.predict(images)
    for i, prediction in enumerate(predictions):
        condition = "Alert" if prediction > 0.5 else "Normal"
        print(f"Frame: {frame_paths[i]} - Condition: {condition}")

# Example usage
video_path = r'C:\Users\Pavan P Kulkarni\Desktop\New folder\video.mp4'
output_folder = r'C:\Users\Pavan P Kulkarni\Desktop\New folder\output'
model_path = r'C:\Users\Pavan P Kulkarni\Desktop\New folder\output.h5'

# Step 1: Extract frames from video
extract_frames(video_path, output_folder)

# Step 2: Load and preprocess the extracted frames
images, frame_paths = load_data(output_folder)

# Step 3: Load a pre-trained model
model = load_model(model_path)

# Step 4: Classify frames and give response
classify_frames(model, images, frame_paths)