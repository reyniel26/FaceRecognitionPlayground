"""
Face Recognizer
----------


Dependencies
----------
The following modules should be installed first 
in order this module run
* cmake
* dlib
* face-recognition
* numpy
* scikit-learn

You can pip the module
>>> pip install cmake
>>> pip install dlib
>>> pip install face-recognition
>>> pip install numpy
>>> pip install scikit-learn # For training of model of Face Recognition
>>> pip install opencv-python # For Face Recognition for video and webcam

"""
import face_recognition
from sklearn import svm
import numpy as np
import os



class FaceRecognition:
    def __init__(self):
        super().__init__()
        #Load the dataset
        self.trained_model = ""

    def face_detect(self,image):
        """
        Detect Face from the image
        ----------

        Return
        ----------
        `NDArray`
            Return Face encoding if there only one detected face

        `bool`
            Return `False` if :
            1) there is no detected face or more than faces
            2) image file not existing
            

        Parameters:
        ----------
        `image` : `str` 
            Image file / path 
        """

        if os.path.exists(image):

            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 1:
                
                return face_recognition.face_encodings(image)[0]

        return False
    
    
    def load_model(self,trained_model):
        """
        Load Model
        ----------

        Load the trained model to be used

        Return
        ----------
        `bool`
            Return `True` if the train model successfully loaded, else Return `False`
            

        Parameters:
        ----------
        `trained_model` : `str` 
            Train model file / path 
        """

        if os.path.exists(trained_model):
            self.trained_model = trained_model
            return True

        return False
    
    def train_model(self,train_dir):
        """
        Train Model
        ----------
        
        Train model from dataset or images

        Parameters:
        ----------
        `train_dir` : `str` 
            path or directory of images
        
        Return
        ----------
        `bool`
            Return `True` if the training is successfull, else Return `False`

        Structure: 
        ----------
        The structure must be the following
        
        >>>    # directory (folder)
        >>>    <train_dir>/ 
        >>>    |   # label (folder)
        >>>    ├── <person1>/ 
        >>>    |   |   # unique file (image) (*.jpg,*.jpeg,*.png)
        >>>    │   ├── <somename1>.jpeg    
        >>>    │   ├── <somename2>.jpeg
        >>>    │   ├── ...
        >>>    ├── <person2>/
        >>>    │   ├── <somename1>.jpeg
        >>>    │   └── <somename2>.jpeg
        >>>    └── ...

        """

        face_encodings = [] # X
        face_labels = []    # Y


        return False


        