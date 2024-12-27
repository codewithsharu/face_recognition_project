import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime
import os
import sys

st.title("Face Recognition Attendance System")

# Add faces section
if st.button("Add New Face"):
    try:
        # Run in a new Python process with terminal
        if sys.platform.startswith('win'):  # For Windows
            result = subprocess.run(["start", "python", "add_faces.py"], shell=True)
        else:  # For Linux/Mac
            result = subprocess.run(["python3", "add_faces.py"], shell=False)
            
        if result.returncode != 0:
            st.error("Error in add_faces.py execution")
        else:
            st.success("Face capture window opened. Please check your camera window.")
    except Exception as e:
        st.error(f"Failed to run add_faces script: {str(e)}")

# Take attendance section
if st.button("Take Attendance"):
    try:
        if sys.platform.startswith('win'):  # For Windows
            result = subprocess.run(["start", "python", "test.py"], shell=True)
        else:  # For Linux/Mac
            result = subprocess.run(["python3", "test.py"], shell=False)
            
        if result.returncode != 0:
            st.error("Error in test.py execution")
        else:
            st.success("Attendance window opened. Please check your camera window.")
    except Exception as e:
        st.error(f"Failed to run test script: {str(e)}")

# Show attendance section
st.header("Today's Attendance")
today = datetime.now().strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{today}.csv"

if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
    st.dataframe(df)
else:
    st.info("No attendance records for today.")