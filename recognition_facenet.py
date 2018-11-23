import os
import cv2
import sys
import dlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance
from collections import Counter

from facenet.src import facenet

def read_rgb(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

class Recognition:
    """
        Class for face recognition using library (https://github.com/davidsandberg/facenet).
    """
    def __init__(self, image_size = (160, 160), data_path = ".\data\ms_faces", embeddings_path = ".\data\data_facenet.json", train = False):
        """
            :param image_size: Image size to which calculates embedding.
            :param data_path: Path to the base of faces. It is assumed that all faces are frontal.
            :param embeddings_path: Path to the base of embeddings.
            :param train: If True than train a new database of embeddings.
        """
        self.image_size = image_size
        facenet.load_model("facenet_model/20180408-102900.pb")
        names = os.listdir(data_path)
        name_face_path = {}
        for name in names:
            faces = os.listdir(os.path.join(data_path, name))
            for face in faces:
                path = os.path.join(data_path, name, face)
                if name in name_face_path.keys():
                    name_face_path[name].append(path)
                else:
                    name_face_path[name] = []
                    name_face_path[name].append(path)
        self.sess = tf.Session()
        if train:
            self.embeddings = dict.fromkeys(name_face_path.keys())
            for name in name_face_path.keys():
                emb = []
                for img_path in name_face_path[name]:
                    image = read_rgb(img_path)
                    emb.append(self.get_embedding(facenet.prewhiten(image)).tolist())
                self.embeddings[name] = emb
            with open(embeddings_path, mode="w") as f:
                json.dump(self.embeddings, f, indent=4)
        else:
            with open(embeddings_path, mode="r") as f:
                self.embeddings = json.load(f)

    def get_embedding(self, img, input_image_size=160):
        """
            :param image: Detected face.
            :return: A list of 128-dimensional face encoding.
         """
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        img = cv2.resize(img, (input_image_size, input_image_size))
        img = img.reshape(-1, input_image_size, input_image_size, 3)
        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
        embedding = self.sess.run(embeddings, feed_dict=feed_dict)
        return embedding

    def find_closest_mean(self, image, unknown_bound=0.85):
        """
            Find the average distance to all people from the base.
            :param image: Detected face.
            :param unknown_bound: If min distance is greater then unknown_bound than face is unknown.
            :return: Unknown or name of the most like person.
        """
        image_embedding = self.get_embedding(facenet.prewhiten(image))
        confidences = {}
        for name, embeddings in self.embeddings.items():
            confidences[name] = np.mean([distance.euclidean(emb, image_embedding) for emb in embeddings])
        print(confidences)
        if min(confidences.values()) > unknown_bound:
            return "unknown"
        else:
            return min(confidences, key=confidences.get)

    def recognize(self, faces):
        """
        :param faces: A list of detected faces which presumably owned by one person.
        :return: Name of the most like person.
        """
        result = list(map(self.find_closest_mean, faces))
        print("result", result)
        return Counter(result).most_common(1)[0][0]

