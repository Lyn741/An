import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(r'E:\New folder\STUDIES\VSC\haarcascade_frontalface_default.xml')
people = ['Albert Einstein', 'Marie Curie']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'E:\New folder\STUDIES\marie-curie_4-5347f.jpg')
if img is None:
    print(f"Error: Could not load image at {img}")
else:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the test image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        # Predict the label of the face
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label: {people[label]} with confidence: {confidence}")

        # Annotate the image with the result
        cv.putText(img, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    cv.imshow('Detected Face', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
