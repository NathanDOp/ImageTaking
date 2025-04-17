import cv2
import easyocr
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime

# Create output directory
output_dir = "detected_texts"
os.makedirs(output_dir, exist_ok=True)

# Load improved YOLOv8 model to use this for any other uses (like the license plate detection).
model = YOLO("yolov8n.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Initialize GUI
root = tk.Tk()
root.title("Text Detection App")

# Video frame
video_label = tk.Label(root)
video_label.pack()

# Text display
text_display = scrolledtext.ScrolledText(root, height=10, width=80)
text_display.pack()

# Global variable to store detected text
detected_text = ""

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    return thresh_image

# Save text and image function
def save_text_and_image(image, text):
    if text.strip():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(output_dir, timestamp)
        os.makedirs(folder_path, exist_ok=True)  # Create a new folder for each capture

        text_filename = os.path.join(folder_path, f"{timestamp}.txt")
        image_filename = os.path.join(folder_path, f"{timestamp}.jpg")

        with open(text_filename, 'w') as f:
            f.write(text)

        cv2.imwrite(image_filename, image)
        messagebox.showinfo("Saved", f"Text saved to {text_filename}\nImage saved to {image_filename}")
    else:
        messagebox.showwarning("No Text", "No text to save.")

# Open folder function
def open_folder():
    folder = os.path.abspath(output_dir)
    os.startfile(folder)

# Capture image function
def capture_image():
    global detected_text
    if hasattr(capture_image, 'last_frame'):
        frame = capture_image.last_frame
    else:
        messagebox.showerror("Error", "No frame captured.")
        return

    # Resize for performance
    small_frame = cv2.resize(frame, (640, 480))

    # Preprocess the image
    preprocessed_frame = preprocess_image(small_frame)

    # Run YOLOv8 inference
    results = model(small_frame, verbose=False, imgsz=640)[0]
    detections = results.boxes.data.cpu().numpy()

    # Draw and OCR on detected regions
    combined_texts = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Ensure the coordinates are within bounds
        if y1 < 0 or y2 > small_frame.shape[0] or x1 < 0 or x2 > small_frame.shape[1]:
            continue

        # Crop the original frame for OCR
        cropped = small_frame[y1:y2, x1:x2]

        # Use EasyOCR to read text from the cropped image
        ocr_results = reader.readtext(cropped, detail=0)

        for text in ocr_results:
            cleaned = text.strip()  # Remove leading/trailing whitespace
            if cleaned:  # Check if the cleaned text is not empty
                combined_texts.append(cleaned)
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small_frame, cleaned, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detected_text = "\n".join(combined_texts)
    text_display.delete(1.0, tk.END)
    text_display.insert(tk.END, detected_text)

    # Save the image and text
    save_text_and_image(small_frame, detected_text)

    # Convert frame for Tkinter
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Show the captured image with detected text
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Store the last frame for capture
    capture_image.last_frame = frame

    # Resize for performance
    small_frame = cv2.resize(frame, (640, 480))

    # Convert frame for Tkinter
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

capture_button = tk.Button(button_frame, text="Capture Image", command=capture_image)
capture_button.pack(side=tk.LEFT, padx=5)

open_button = tk.Button(button_frame, text="Open Folder", command=open_folder)
open_button.pack(side=tk.LEFT, padx=5)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Start processing frames for live video feed
process_frame()
root.mainloop()

# Release video capture
cap.release()
cv2.destroyAllWindows()