import cv2
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime
from PIL import ImageGrab
import subprocess
import easyocr  # Import EasyOCR

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify other languages as needed


# Function to detect text using EasyOCR
def detect_text(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Use EasyOCR to detect text
    results = reader.readtext(image)

    # Extract the detected text
    detected_text = ""
    for (bbox, text, prob) in results:
        detected_text += f"{text}\n"  # Append each detected text line

    return detected_text.strip()  # Return the detected text


# Function to take a picture and detect text
def take_picture():
    ret, frame = cap.read()
    if ret:
        if not os.path.exists('captured_images'):
            os.makedirs('captured_images')

        filename = f"captured_images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Success", f"Image saved as {filename}")

        # Detect text from the image
        detected_text = detect_text(filename)

        # Save the detected text to a text file
        text_filename = f"captured_images/extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_filename, 'w') as text_file:
            text_file.write(detected_text)

        messagebox.showinfo("Text Extracted", f"Extracted text saved as {text_filename}")
    else:
        messagebox.showerror("Error", "Failed to capture image")


# Function to take a screenshot
def take_screenshot():
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')

    filename = f"captured_images/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    messagebox.showinfo("Success", f"Screenshot saved as {filename}")


# Function to open the folder containing images
def open_folder():
    folder_path = os.path.abspath('captured_images')
    subprocess.Popen(f'explorer "{folder_path}"')


# Function to close the webcam
def close_camera():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create the main window
root = tk.Tk()
root.title("Webcam Capture")

# Create a button to take a picture
capture_button = tk.Button(root, text="Capture Image", command=take_picture)
capture_button.pack(pady=10)

# Create a button to take a screenshot
screenshot_button = tk.Button(root, text="Take Screenshot", command=take_screenshot)
screenshot_button.pack(pady=10)

# Create a button to open the folder
open_folder_button = tk.Button(root, text="Open Images Folder", command=open_folder)
open_folder_button.pack(pady=10)

# Create a button to exit the application
exit_button = tk.Button(root, text="Exit", command=close_camera)
exit_button.pack(pady=20)

# Start the GUI loop
root.protocol("WM_DELETE_WINDOW", close_camera)
root.mainloop()