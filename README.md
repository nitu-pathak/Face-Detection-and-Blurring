# Face-Detection-and-Blurring

The code detects faces in the images and webcam using Haar Cascade Classifier for frontal face detection with openCV.

1. Face_detection_and_blurring.py detects faces in the images using Haar Cascade Classifier for frontal face detection with openCV and blurs the faces

2. video_face_blurring.py detects faces in the webcam frames using Haar Cascade Classifier for frontal face detection with openCV and blurs the faces in the video

3. video_background_blurring.py detects faces in the webcam frames using Haar Cascade Classifier for frontal face detection with openCV and blurs the background (except the face) in the video

The CascadeClassifier has many parameters like scaleFactor, minNeighbors, minSize, maxSize. The parameter tuning was done to obtain the best results.
Also, the results are as saved as image.
