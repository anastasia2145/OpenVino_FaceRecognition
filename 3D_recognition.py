import os
import cv2
import json
import numpy as np
import pandas as pd
from time import time
import scipy.io as sio
from scipy.spatial.distance import directed_hausdorff

import matplotlib.pylab as plt

from PRNet.api import PRN


def read_rgb(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def show(image):
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt.show()


class Recognition:
    """
       Class for face recognition using library (https://github.com/YadiraF/PRNet).
    """
    def __init__(self, image_size=(96, 96), data_path=".\data\ms_faces", embeddings_path=".\data\pos.json", train=False):
        self.image_size = image_size
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU number, -1 for CPU
        self.prn = PRN(is_dlib=True)
        if train:
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

            self.embeddings = dict.fromkeys(name_face_path.keys())
            for name in name_face_path.keys():
                name_vertices = []
                for img_path in name_face_path[name]:
                    image = read_rgb(img_path)
                    image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
                    n, m, _ = image.shape
                    pos = self.prn.process(input=image, image_info=np.array([0, m, 0, n]))
                    vertices = self.prn.get_vertices(pos)
                    name_vertices.append(vertices.tolist())
                self.embeddings[name] = name_vertices
            with open(embeddings_path, mode="w") as f:
                json.dump(self.embeddings, f, indent=4)
        else:
            with open(embeddings_path, mode="r") as f:
                self.embeddings = json.load(f)

    def get_embedding(self, image):
        """
            :param image: Detected face.
            :return: Array of vertices with shape=(43867,3).
        """
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        n, m, _ = image.shape
        pos = self.prn.process(image, image_info=np.array([0, m, 0, n]))
        return self.prn.get_vertices(pos)

    # def hausdorff_distance(self, embedding1, embedding2):
    #     return max(directed_hausdorff(embedding1, embedding2), directed_hausdorff(embedding2, embedding1))[0]

    def hausdorff_distance(self, embeddings, img_emb):
        """
            Compute hausdorff_distance between set of embeddings and current embedding.
            :param embeddings: Set of embeddings with which we will compare.
            :param img_emb: Current embedding to compare with.
            :return: A list of distances between embeddings and current image embedding.
        """
        dists = []
        for emb in embeddings:
            dists.append(max(directed_hausdorff(emb, img_emb), directed_hausdorff(img_emb, emb))[0])
        return dists

    def find_closest_mean(self, image):
        """
             Find the average distance to all people from the base.
             :param image: Detected face.
             :return: Name of the most like person.
         """
        image_embedding = self.get_embedding(image)
        confidences = {}
        for name, embeddings in self.embeddings.items():
            confidences[name] = np.mean(self.hausdorff_distance(embeddings, image_embedding))
        print("result:", min(confidences, key=confidences.get))
        print(confidences)
        return min(confidences, key=confidences.get)


if __name__ == "__main__":
    rec = Recognition(train=False)
    img = read_rgb("paskar_1.jpg")
    n,m,_ = img.shape
    image_embedding = rec.get_embedding(img)

    start = time()
    image_embedding = rec.get_embedding(img)
    print("Time to get embedding:", time() - start)

    start = time()
    rec.find_closest_mean(img)
    print("Time to recognition:", time() - start)

    test_path = "data/test"
    result = []
    for file in os.listdir(test_path):
        img = read_rgb(os.path.join(test_path, file))
        result.append([file, rec.find_closest_mean(img)])
    df = pd.DataFrame(result, columns=["name", "mean res"])
    print(df)