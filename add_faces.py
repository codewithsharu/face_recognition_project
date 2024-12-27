import cv2
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize paths
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def capture_faces():
    name = input("Enter name of person: ").strip()
    if not name:
        print("Name cannot be empty")
        return
    
    # Initialize as lists
    faces_data = []
    names = []
    
    # Load existing data if available
    if os.path.exists(os.path.join(DATA_DIR, "faces_data.pkl")):
        with open(os.path.join(DATA_DIR, "faces_data.pkl"), 'rb') as f:
            existing_faces = pickle.load(f)
            # Ensure existing data is in correct shape
            if isinstance(existing_faces, np.ndarray):
                faces_data = existing_faces.tolist()
        with open(os.path.join(DATA_DIR, "names.pkl"), 'rb') as f:
            names = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print("Capturing faces... Press 'q' to quit")
    
    while count < 100:
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
        
        cv2.putText(frame, f"Captured: {count}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (50, 50))
            
            # Ensure consistent shape: flatten to 1D array of size 2500 (50x50)
            face_array = face.reshape(2500)
            faces_data.append(face_array)
            names.append(name)
            count += 1
            print(f"Picture {count}/50 taken")
            
            cv2.waitKey(100)
            
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the data
    if count > 0:
        try:
            # Convert to numpy array with explicit shape
            faces_data_array = np.array(faces_data, dtype=np.float32)
            with open(os.path.join(DATA_DIR, "faces_data.pkl"), 'wb') as f:
                pickle.dump(faces_data_array, f)
            with open(os.path.join(DATA_DIR, "names.pkl"), 'wb') as f:
                pickle.dump(names, f)
            print(f"Data captured and saved successfully for {name}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            print("Please try again")
    else:
        print("No faces were captured. Please try again.")

if __name__ == "__main__":
    capture_faces()