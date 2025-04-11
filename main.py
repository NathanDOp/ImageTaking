import cv2
import easyocr
import torch
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime
import re

# Create output directory
output_dir = "detected_texts"
os.makedirs(output_dir, exist_ok=True)

# Load improved YOLOv8 model (trained on license plates or text)
model = YOLO("yolov8n.pt")  # Replace with possibly yolov8n_text.pt

# Load EasyOCR with English and digit support
reader = easyocr.Reader(['en'], gpu=True)

# Initialize GUI
root = tk.Tk()
root.title("Live Text Detection App")

# Video frame
video_label = tk.Label(root)
video_label.pack()

# Text display
text_display = scrolledtext.ScrolledText(root, height=10, width=80)
text_display.pack()

# Global variable to store detected text
detected_text = ""

# Save text function
def save_text():
    global detected_text
    if detected_text.strip():
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(detected_text)
        messagebox.showinfo("Saved", f"Text saved to {filepath}")
    else:
        messagebox.showwarning("No Text", "No text to save.")

# Open folder function
def open_folder():
    folder = os.path.abspath(output_dir)
    os.startfile(folder)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

save_button = tk.Button(button_frame, text="Save Text", command=save_text)
save_button.pack(side=tk.LEFT, padx=5)

open_button = tk.Button(button_frame, text="Open Folder", command=open_folder)
open_button.pack(side=tk.LEFT, padx=5)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Store last text to avoid repeated updates
last_detected_text = ""

# Regex pattern for license plates (simple example)
plate_regex = re.compile(r'^[A-Z0-9]{4,10}$')

def process_frame():
    global detected_text, last_detected_text
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    # Resize for performance
    small_frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 inference
    results = model(small_frame, verbose=False, imgsz=640)[0]
    detections = results.boxes.data.cpu().numpy()

    # Draw and OCR on detected regions
    combined_texts = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped = small_frame[y1:y2, x1:x2]
        ocr_results = reader.readtext(cropped, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for bbox, text, confidence in ocr_results:
            cleaned = text.strip().replace(" ", "")
            if confidence > 0.5 and (plate_regex.match(cleaned) or len(cleaned) > 3):
                combined_texts.append(cleaned)
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small_frame, cleaned, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detected_text = "\n".join(combined_texts)
    if detected_text != last_detected_text:
        text_display.delete(1.0, tk.END)
        text_display.insert(tk.END, detected_text)
        last_detected_text = detected_text

    # Convert frame for Tkinter
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)

# Start processing
process_frame()
root.mainloop()

# Release video capture
cap.release()
cv2.destroyAllWindows()
