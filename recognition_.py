import os
import cv2
import sys
import json
import time
import numpy as np
import pandas as pd
import face_recognition
import matplotlib.pylab as plt
from collections import Counter



def read_rgb(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def show(image):
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


class Recognition:
    """
    Class for face recognition using library (https://github.com/ageitgey/face_recognition).
    """
    def __init__(self, image_size=(96, 96), data_path=".\data\ms_faces", embeddings_path=".\data\data.json", train=False):
        """
        :param image_size: Image size to which calculates embedding.
        :param data_path: Path to the base of faces.
        :param embeddings_path: Path to the base of embeddings.
        :param train: If True than train a new database of embeddings.
        """
        self.image_size = image_size
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

        if train:
            self.embeddings = dict.fromkeys(name_face_path.keys())
            for name in name_face_path.keys():
                emb = []
                for img_path in name_face_path[name]:
                    image = read_rgb(img_path)
                    image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
                    n, m, _ = image.shape
                    emb.append(face_recognition.face_encodings(image, known_face_locations=[[0, 0, n, m]])[0].tolist())
                self.embeddings[name] = emb
            with open(embeddings_path, mode="w") as f:
                json.dump(self.embeddings, f, indent=4)
        else:
            with open(embeddings_path, mode="r") as f:
                self.embeddings = json.load(f)

    def get_embedding(self, image):
        """
        :param image: Detected face.
        :return: A list of 128-dimensional face encoding.
        """
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        n, m, _ = image.shape
        return face_recognition.face_encodings(image, known_face_locations=[[0, 0, n, m]])[0]

    def find_closest_mean(self, image, unknown_bound=0.8):
        """
        Find the average distance to all people from the base.
        :param image: Detected face.
        :param unknown_bound: If min distance is greater then unknown_bound than face is unknown.
        :return: Unknown or name of the most like person.
        """
        image_embedding = self.get_embedding(image)
        confidences = {}
        for name, embeddings in self.embeddings.items():
            confidences[name] = face_recognition.face_distance(embeddings, image_embedding).mean()
        if min(confidences.values()) > unknown_bound:
            return "unknown"
        else:
            return min(confidences, key=confidences.get)

    def find_closest_vote(self, image, unknown_bound=0.2, tolerance=0.4):
        """
        :param image: Detected face.
        :param unknown_bound: If the percentage of voters for is less than unknown_bound then face is unknown.
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict.
        :return: Unknown or name of the most like person.
        """
        image_embedding = self.get_embedding(image)
        confidences = {}
        for name, embeddings in self.embeddings.items():
            votes = face_recognition.compare_faces(embeddings, image_embedding, tolerance=tolerance)
            confidences[name] = sum(votes) / len(votes)
        if max(confidences.values()) < unknown_bound:
            return "unknown"
        else:
            return max(confidences, key=confidences.get)

    def recognize(self, faces, method):
        """
        :param faces: A list of detected faces which presumably owned by one person.
        :param method: Shows which method used to compare faces. Can be "mean" or "vote".
        :return: Name of the most like person
        """
        if method == "mean":
            result = list(map(self.find_closest_mean, faces))
            return Counter(result).most_common(1)[0][0]
        elif method == "vote":
            result = list(map(self.find_closest_vote, faces))
            return Counter(result).most_common(1)[0][0]


if __name__ == "__main__":
    rec = Recognition(train=True)
    img = read_rgb("anikin_5.jpg")
    n,m,_ = img.shape
    image_embedding = face_recognition.face_encodings(img, known_face_locations=[[0, 0, n, m]])[0]
    img_1 = read_rgb("anikin_1.jpg")
    img_2 = read_rgb("anikin_5.jpg")
    img_3 = read_rgb("anikin_7.jpg")
    print(rec.recognize(faces=[img_1, img_2, img_3], method="mean"))
    test_path = "data/test"
    result = []
    for file in os.listdir(test_path):
        img = read_rgb(os.path.join(test_path, file))
        result.append([file, rec.find_closest_mean(img), rec.find_closest_vote(img)])
    df = pd.DataFrame(result, columns=["name", "mean res", "vote res"])
    print(df)
