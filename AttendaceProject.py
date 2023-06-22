import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

# Set the path to the directory containing the images
path = 'Images'

# Create empty lists to store images and corresponding class names
images = []
classNames = []

# Retrieve the list of files in the specified directory
myList = os.listdir(path)
print(myList)

# Iterate through each file in the directory
for c1 in myList:
    # Read the image using OpenCV
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)  # Append the image to the list
    # Extract the class name from the file name without the extension
    classNames.append(os.path.splitext(c1)[0])
print(classNames)

# Function to encode the faces in the given list of images
def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        encode = face_recognition.face_encodings(img)[0]  # Encode the face in the image
        encodingList.append(encode)  # Append the encoding to the list
    return encodingList

# Function to mark attendance in the CSV file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0].strip())
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')
        print(myDataList)

# Encode the faces in the provided images
encodeListKnown = findEncodings(images)
print("Encoding Complete")

filename = 'Attendance.csv'
if os.path.isfile(filename):
    print(f"The file '{filename}' already exists.")
else:
    print(f"The file '{filename}' does not exist. Creating a new file...")
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header if needed
        writer.writerow(['Name', 'Time'])
    print(f"New file '{filename}' created successfully.")

# Encode the faces in the provided images
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Open the webcam for video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Read a frame from the video capture
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the frame for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format
    faceCurFrame = face_recognition.face_locations(imgS)  # Locate faces in the frame
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)  # Encode the faces in the frame

    for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare the faces with known encodings
        facdis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Calculate the face distance
        #print(facdis)
        matchIndex = np.argmin(facdis)  # Find the index of the best match based on face distance

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # Retrieve the name corresponding to the best match
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale the face location coordinates
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the face
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Draw a filled rectangle for the name label
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Add the name label
            markAttendance(name)  # Mark the attendance in the CSV file
    cv2.imshow('WebCam', img)  # Display the processed frame
    cv2.waitKey(1)
