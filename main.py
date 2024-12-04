import cv2
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime
from PIL import ImageGrab  # Importing ImageGrab for screenshots
import subprocess  # For opening the folder

# Function to take a picture
def take_picture():
    # Capture from the webcam
    ret, frame = cap.read()
    if ret:
        # Create a folder to save images if it doesn't exist
        if not os.path.exists('captured_images'):
            os.makedirs('captured_images')

        # Create a filename with the current timestamp
        filename = f"captured_images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Save the image
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Success", f"Image saved as {filename}")
    else:
        messagebox.showerror("Error", "Failed to capture image")

# Function to take a screenshot
def take_screenshot():
    # Create a folder to save screenshots if it doesn't exist
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')

    # Create a filename with the current timestamp
    filename = f"captured_images/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Take a screenshot
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    messagebox.showinfo("Success", f"Screenshot saved as {filename}")

# Function to open the folder containing images
def open_folder():
    folder_path = os.path.abspath('captured_images')
    subprocess.Popen(f'explorer "{folder_path}"')  # For Windows
    # For macOS, you can use: subprocess.Popen(['open', folder_path])
    # For Linux, you can use: subprocess.Popen(['xdg-open', folder_path])

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