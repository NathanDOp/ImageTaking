import cv2
import easyocr
import os
import json
import tkinter as tk
from tkinter import messagebox, StringVar, Toplevel
from tkinter import scrolledtext
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime

# Create output directory
output_dir = "detected_texts"
os.makedirs(output_dir, exist_ok=True)

# Load improved YOLOv8 model
model = YOLO("yolov8n.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Load pre-saved text from a JSON file
def load_pre_saved_text(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# Load pre-saved texts at the start of your program
pre_saved_texts = load_pre_saved_text('pre_saved_texts.json')

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
cap = None  # Initialize camera variable

# Function to preprocess the image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return thresh_image

# Save text and image function
def save_text_and_image(image, text, corrected_text=None):
    if text.strip():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(output_dir, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        text_filename = os.path.join(folder_path, f"{timestamp}.txt")
        image_filename = os.path.join(folder_path, f"{timestamp}.jpg")

        with open(text_filename, 'w') as f:
            f.write(text)

        cv2.imwrite(image_filename, image)

        # Save corrected text if provided
        if corrected_text:
            corrected_text_filename = os.path.join(folder_path, f"{timestamp}_corrected.txt")
            with open(corrected_text_filename, 'w') as f:
                f.write(corrected_text)

            # Update pre-saved texts
            pre_saved_texts[corrected_text] = corrected_text  # Store corrected text
            with open('pre_saved_texts.json', 'w') as f:
                json.dump(pre_saved_texts, f)

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

    # Run YOLOv8 inference
    results = model(small_frame, verbose=False, imgsz=640)[0]
    detections = results.boxes.data.cpu().numpy()

    # Draw and OCR on detected regions
    combined_texts = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if y1 < 0 or y2 > small_frame.shape[0] or x1 < 0 or x2 > small_frame.shape[1]:
            continue

        cropped = small_frame[y1:y2, x1:x2]  # Fixed typo here
        ocr_results = reader.readtext(cropped, detail=0)

    for text in ocr_results:
        cleaned = text.strip()
        if cleaned:
            # Check against pre-saved texts for potential corrections
            if cleaned in pre_saved_texts:
                cleaned = pre_saved_texts[cleaned]  # Use corrected version
            combined_texts.append(cleaned)
            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(small_frame, cleaned, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detected_text = "\n".join(combined_texts)
    text_display.delete(1.0, tk.END)
    text_display.insert(tk.END, detected_text)

    # Show preview window for editing text
    show_preview(small_frame, detected_text)

def show_preview(image, text):
    preview_window = Toplevel(root)
    preview_window.title("Preview and Edit Text")

    # Create a label for the image
    img_label = tk.Label(preview_window)
    img_label.pack()

    # Convert frame for Tkinter
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)

    # Create a text box for editing detected text
    text_var = StringVar(value=text)
    text_entry = tk.Entry(preview_window, textvariable=text_var, width=80)
    text_entry.pack(pady=10)

    # Save button
    save_button = tk.Button(preview_window, text="Save", command=lambda: save_and_close(preview_window, image, text, text_var.get()))
    save_button.pack(pady=5)

def save_and_close(preview_window, image, original_text, corrected_text):
    save_text_and_image(image, original_text, corrected_text)
    preview_window.destroy()  # Close the preview window after saving

def list_cameras():
    index = 0
    camera_list = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camera_list.append(index)
        cap.release()
        index += 1
    return camera_list

# Function to switch camera
def switch_camera(selected_camera):
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = cv2.VideoCapture(selected_camera)
    cap.set(cv2.CAP_PROP_FPS, 30)
    process_frame()

# Create camera selection menu
camera_menu = StringVar(root)
camera_menu.set("Select Camera")
camera_options = list_cameras()
camera_dropdown = tk.OptionMenu(root, camera_menu, *camera_options, command=switch_camera)
camera_dropdown.pack(pady=5)

# Process frame function
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

    capture_image.last_frame = frame
    small_frame = cv2.resize(frame, (640, 480))

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
if cap is not None:
    cap.release()
cv2.destroyAllWindows()