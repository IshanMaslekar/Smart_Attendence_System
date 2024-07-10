import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and their encodings
known_faces = []
known_names = []

def load_known_face(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_faces.append(encoding[0])
        known_names.append(name)

load_known_face("faces/Rohit.jpg", "Rohit")
load_known_face("faces/Virat.jpg", "Virat")
load_known_face("faces/Ishan.jpg", "Ishan")
load_known_face("faces/img_1.png", "Piyush")
load_known_face("faces/img.jpg", "Vedant")

# Open the video capture
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []
face_encodings = []
students_present = set()

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create a CSV writer
csv_filename = f"{current_date}.csv"
with open(csv_filename, "w+", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_faces, encoding)
            name = "Unknown"

            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]

            # Draw a box and label around the face
            cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 0.5, (255, 255, 255), 1)

            # Record attendance if the person is a known student
            if name in known_names:
                students_present.add(name)
                current_time = now.strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])

        # Display the frame with face recognition results
        cv2.imshow("Attendance", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Print the list of students present
print("Students Present:")
for student in students_present:
    print(student)
