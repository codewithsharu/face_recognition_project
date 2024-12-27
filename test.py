import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tkinter import messagebox
import tkinter as tk

def show_popup(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Attendance", message)
    root.destroy()

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")
    
    attendance_dir = "Attendance"
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
        
    filename = f"Attendance/Attendance_{date}.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])
    
    if not df['Name'].str.contains(name).any():
        df = pd.concat([df, pd.DataFrame([[name, time]], columns=['Name', 'Time'])], ignore_index=True)
        df.to_csv(filename, index=False)
        show_popup(f"Attendance marked for {name}")
        return True
    else:
        show_popup(f"Attendance already marked for {name}")
        return False

def recognize_faces():
    # Load the training data
    try:
        with open("data/faces_data.pkl", 'rb') as f:
            faces_data = pickle.load(f)
        with open("data/names.pkl", 'rb') as f:
            names = pickle.load(f)
            
        print(f"Loaded training data: {faces_data.shape} faces, {len(names)} names")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Prepare the data
    if len(faces_data) == 0 or len(names) == 0:
        print("No training data available. Please add faces first.")
        return
        
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_data, names)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    current_name = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (50, 50))
            face = face.flatten().reshape(1, -1)
            
            # Predict
            name = knn.predict(face)[0]
            confidence = knn.predict_proba(face).max()
            current_name = name
            
            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Add instruction text
            cv2.putText(frame, "Press 'y' to mark attendance", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Check for key presses
        if key == ord('q'):
            break
        elif key == ord('y') and current_name is not None:
            mark_attendance(current_name)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()

