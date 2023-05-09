import face_recognition
import csv
import numpy as np
import cv2
from datetime import datetime
import os

# take input from camera
video_capture = cv2.VideoCapture(0)

imgArray=[]
imgEncodingArray=[]
known_faces_encoding=[]
known_faces_names =[]
i=0
# print(os.listdir("./photos"))

for images in os.listdir("./photos"):
  if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
    imgArray.append(face_recognition.load_image_file("photos/"+images))
    imgEncodingArray.append( face_recognition.face_encodings(imgArray[i])[0])
    known_faces_encoding.append(imgEncodingArray[i])
    known_faces_names.append(os.path.splitext(images)[0])

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
        if name in students:
          students.remove(name)
          print(students)
          cv2.putText(frame, 'Present', (250, 400), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (14, 157, 87), 2, cv2.LINE_AA)
          cur_time = now.strftime("%H-%M-%S")
          writer_.writerow([name, cur_time])
      else:
        cv2.putText(frame, 'Student Not Found', (250, 400), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (228, 152, 28), 2, cv2.LINE_AA)


  cv2.imshow("FACIAL ATTENDANCE SYSTEM", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
f.close()
