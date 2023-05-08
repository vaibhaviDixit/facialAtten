import face_recognition
import csv
import numpy as np
import os
import cv2
from datetime import datetime

# take input from camera
video_capture = cv2.VideoCapture(0)

ankit_img = face_recognition.load_image_file("photos/Ankit.jpg")
ankit_encoding = face_recognition.face_encodings(ankit_img)[0]

rakesh_img = face_recognition.load_image_file("photos/Rakesh.jpg")
rakesh_encoding = face_recognition.face_encodings(rakesh_img)[0]

riya_img = face_recognition.load_image_file("photos/riya.jpg")
riya_encoding = face_recognition.face_encodings(riya_img)[0]

known_faces_encoding = [ankit_encoding, rakesh_encoding, riya_encoding]
known_faces_names = ["Ankit", "Rakesh", "Riya"]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

# date and time
now = datetime.now()
cur_date = now.strftime("%d-%m-%Y")

# open csv file
f = open(cur_date + ".csv", 'a', newline='')
writer_ = csv.writer(f)

while True:
  _,frame = video_capture.read()
  small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
  rgb_frame = small_frame[:, :, ::-1]

  if s:
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
      match = face_recognition.compare_faces(known_faces_encoding,
                                             face_encoding)
      name = ""
      face_distance = face_recognition.face_distance(known_faces_encoding,
                                                     face_encoding)
      best_index = np.argmin(face_distance)

      if match[best_index]:
        name = known_faces_names[best_index]

      face_names.append(name)
      if name in known_faces_names:
        students.remove(name)
        print(students)
        cur_time = now.strftime("%H-%M-%S")
        writer_.writerow([name, cur_time])

  cv2.imshow("FACIAL ATTENDANCE SYSTEM", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
f.close()
