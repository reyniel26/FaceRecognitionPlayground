from facerecognizer import FaceRecognition
import os
import time

algo = "knn"
threshold = 0.6

"""

Threshold 

0.2 == Atleast 20% face distance and 80% match of face encodings
0.4 == Atleast 40% face distance and 60% match of face encodings
0.6 == Atleast 60% face distance and 40% match of face encodings

The advisable is 0.4718 to 0.6 according to the author of face_recognition module
Lower the threshold makes the comparison more strict 

"""

facerecognition = FaceRecognition()

# Train
print("Training algorithm")
facerecognition.train_model("train_dir","models",algo=algo)

# Load
print("Load the trained algorithm")
facerecognition.load_model("models/trained_"+algo+"_model.clf")

# Test
print("Test the trained algorithm")
images = os.listdir("test_dir")
for image in images:
    start = time.time()
    image_path = os.path.join("test_dir",image)
    prediction = facerecognition.predict(image_path, algo = algo ,threshold = threshold)
    end = time.time()
    print("Prediction: {} , SPEED: {}".format(prediction,str(end - start)))
