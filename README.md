WELCOME!

Developer: Nathan D'Agostino

This was a Senior Engineering project made for the Lee County Sheriff in Estero Florida.
The project entailed the creation of a OCR (Optical Character Recognition) software to recognize images and record those images down and save them in their own individual file.
This project is still currently as of 5/1/2025 in Alpha stages and requires the following
    -  A trained .pt model of USA license plates as the current model is using a rectangle/square identifier model. (Current is yolov8n.pt)
    -  Timestamp coordination on the images/folder storage.
    -  Have the 'Open File' button work on linux (Works only on windows)
    -  Find a way to have both live and single-frame detection (Currently only in single frame as the live detection would cause severe amounts of lag)
    -  Find a way to sync this program with the "Auterion Skynode" interface.
    -  Add a way to zoom in and zoom out.

Please make sure your Python IDE has:
  - tkinter to import messagebox, StringVar, Toplevel, scrolledtext
  - PIL to import Image, ImageTk
  - ultralytics to import YOLO
  - easyocr

To Run The Software (As of Alpha Version 0.0.1)
  - Open Pycharm or any Python-Running IDE
  - Import the code and ensure all libraries and frameworks are installed.
  - Run code as normal.

  - When code is running you are free to use it as you desire.
