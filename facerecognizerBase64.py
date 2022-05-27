"""


Images Table
- id : int
- base64image : LargeBinary / Bitea 

Dataset Table
- id 
- image_id : FK id Images Table
- user_id : FK user Table

In query,
- Inner join 
- base64image AS base64image
- user_id AS label



"""


from facerecognizer import FaceRecognition
import os
import os.path
import base64
import face_recognition
from sklearn import svm
import numpy as np
import joblib
import math
from sklearn import neighbors
import pickle
import io


class Base64FaceRecognition(FaceRecognition):

    def __init__(self):
        super().__init__()
    

    #Override Train
    def train_model(self,train_datasets, model_dir:str="", algo:str = "knn" ):
        """
        Train Model
        ----------
        
        Train model from dataset or images

        Parameters:
        ----------
        `train_datasets` 
            must be table with the colums of (label:`str`, base64image:`bytes`)
        
        `model_dir` : `str` 
            path or directory where the model to be save
        
        `algo` : `str` 
            Set algo to be used for training. The default algorithm is `knn`. The available algorithm are `svm` and `knn`
        
        Return
        ----------
        `bool`
            Return `True` if the training is successfull, else Return `False`

        """
        if algo == "svm":
            return self._train_model_svm(train_datasets,model_dir)
        else:
            return self._train_model_knn(train_datasets, model_save_path=model_dir ,n_neighbors=2)

    def _train_model_svm(self,train_datasets , model_dir:str="" ):
        """
        Train Model using SVM algorithm
        ----------
        
        Train model from dataset or images

        Parameters:
        ----------
        `train_datasets` 
            must be table with the colums of (label:`str`, base64image:`bytes`)
        
        `model_dir` : `str` 
            path or directory where the model to be save
        
        Return
        ----------
        `bool`
            Return `True` if the training is successfull, else Return `False`

        """

        face_encodings = [] # X
        face_labels = []    # Y

        for data in train_datasets:

            decoded = base64.decodebytes(data.base64image)
            face =  face_recognition.load_image_file(io.BytesIO(decoded))
            face_locations = face_recognition.face_locations(face)

            if len(face_locations) == 1:
                face_code  = face_recognition.face_encodings(face)[0]
                face_encodings.append(face_code)
                face_labels.append(data.label)
            else:
                print(str(data.label) + " was skipped and can't be used for training")

        # If empty
        if not (face_encodings and face_labels):
            return False

        # Create and train the SVC classifier
        clf = svm.SVC(gamma='scale')
        clf.fit(face_encodings,face_labels)

        model_name = os.path.join(model_dir,"trained_svm_model.clf")
        joblib.dump(clf,model_name)

        return True
        
    def _train_model_knn(self,train_datasets, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Train Model using KNN algorithm
        ----------
        
        Trains a k-nearest neighbors classifier for face recognition.
        :param `train_datasets` must be table with the colums of (label:`str`, base64image:`bytes`)

        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.

        """

        X = []
        y = []

        # Loop through each person in the training set
        for data in train_datasets:
            decoded = base64.decodebytes(data.base64image)
            image =  face_recognition.load_image_file(io.BytesIO(decoded))

            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(data.label, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(data.label)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            model_save_path = os.path.join(model_save_path,"trained_knn_model.clf")
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf

    #Overide Predict
    def predict(self,base64image:str, algo:str = "knn", threshold=0.6):
        """
        Predict 
        ----------

        Predict the label for the image based on the trained model

        Parameters:
        ----------
        `image` : `str` 
            path or file image
        
        `algo` : `str` 
            Set algo to be used for training. The default algorithm is `knn`. The available algorithm are `svm` and `knn`
        
        `threshold`:`float`
            Set threshold. Threshold is the maximum face distance and the face distance should be lower than that to be accepted
            However `svm` doesnt uses threshold, its advisable for svm to have unknown labeled dataset for training to show the unknown.
        
        See more:
        ----------

        * More about threshold, you can visit the link  :ref:`Calculating Accuracy as a Percentage <https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage>`.

        """
        if algo == "svm":
            return self._predict_svm(base64image)
        else:
            return self._predict_knn(base64image,distance_threshold=threshold)

    def _predict_svm(self,base64image:str):
        
        if not self.trained_model and os.path.exists(self.trained_model):
            return []

        # unknown_image = face_recognition.load_image_file(image)
        decoded = base64.decodebytes(base64image)
        unknown_image =  face_recognition.load_image_file(io.BytesIO(decoded))

        unknown_image_enc = face_recognition.face_encodings(unknown_image)[0]
        
        svm_model = joblib.load(self.trained_model)

        return svm_model.predict([unknown_image_enc])
    
    def _predict_knn(self,base64image:str,knn_clf=None, distance_threshold=0.6):
        """
        Recognizes faces in given image using a trained KNN classifier
        :param image: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
            of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """


        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            if not self.trained_model and os.path.exists(self.trained_model):
                raise Exception("The model is not loaded. Must supply knn classifier either thourgh knn_clf or load_model()")

            with open(self.trained_model, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        # X_img = face_recognition.load_image_file(image)
        decoded = base64.decodebytes(base64image)
        X_img = face_recognition.load_image_file(io.BytesIO(decoded))
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) != 1:
            raise Exception("Must have one and only one face in the image")

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        percentage = [round((1-closest_distances[0][i][0])*100, 2) for i in range(len(X_face_locations))]

        """
        threshold here is used to check if the face distance is lower or equal to the threshold
        """

        # Predict classes and remove classifications that aren't within the threshold
        return [ (pred,perc) if rec else ("unknown",perc) for pred, loc, rec,perc in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches,percentage)]

