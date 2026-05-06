import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

# ----------------------- LANE DETECTION FUNCTION -----------------------
def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros_like(edges)
    height, width = mask.shape
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)

    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)

    result = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
    return result

# ----------------------- LIVE CAMERA FUNCTION -----------------------
def live_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        lane_frame = detect_lane(frame)
        cv2.imshow("Live Lane Detection - Press Q to Quit", lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------- VIDEO FILE FUNCTION -----------------------
def video_from_file():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not filepath:
        return

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lane_frame = detect_lane(frame)
        cv2.imshow("Video Lane Detection - Press Q to Quit", lane_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------------- EXIT FUNCTION -----------------------
def close_app():
    try:
        cv2.destroyAllWindows()
    except:
        pass
    root.quit()
    root.destroy()
    os._exit(0)  # Force kill the process if anything still runs

# ----------------------- GUI SETUP -----------------------
root = tk.Tk()
root.title("ROAD LANE DETECTOR")
root.geometry("400x300")
root.resizable(False, False)
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="ROAD LANE DETECTOR", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333")
title.pack(pady=20)

btn1 = tk.Button(root, text="📷 Start Live Camera", command=live_camera,
                 font=("Arial", 12), bg="#4CAF50", fg="white", width=25)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="📁 Upload Video File", command=video_from_file,
                 font=("Arial", 12), bg="#2196F3", fg="white", width=25)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="❌ Close", command=close_app,
                 font=("Arial", 12), bg="#f44336", fg="white", width=25)
btn3.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", close_app)  # Handle X button
root.mainloop()
