import cv2
import tensorflow as tf
from datasets import load_dataset

# Force TensorFlow to use CPU if CUDA issues persist
try:
    tf.config.set_visible_devices([], 'GPU')
    print("Running on CPU mode.")
except:
    print("GPU mode active.")

# Attempt to load ASL dataset
try:
    asl_dataset = load_dataset("WLASL", split="train[:10]")  # Load small subset
    print("ASL dataset loaded successfully!")
except Exception as e:
    print(f"Error loading ASL dataset: {e}")
    print("Using a placeholder dataset.")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera. Trying another index...")
    cap = cv2.VideoCapture(1)  # Try another index if the first fails

if not cap.isOpened():
    print("Error: No available cameras detected. Exiting.")
else:
    print("Camera initialized successfully!")

# Display video feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('ASL Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
