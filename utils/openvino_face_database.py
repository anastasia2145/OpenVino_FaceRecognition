from __future__ import print_function

import json
import os
import cv2
import numpy as np
from time import time
import inspect

from scipy import spatial
from openvino.inference_engine import IENetwork, IEPlugin
from utils.face_aligner import FaceAligner


class OpenvinoFaceDataBase:
    def __init__(self, log, face_samples, cpu_lib="./lib/cpu_extension.dll",
                 detector_xml="./models/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml",
                 landmarks_xml="./models/intel_models/landmarks-regression-retail-0001/FP32/landmarks-regression-retail-0001.xml",
                 features_xml="./models/intel_models/face-reidentification-retail-0001/FP32/face-reidentification-retail-0001.xml",
                 age_gender_xml="./models/intel_models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
                 detection_threshold=0.8, train=True, data_path="./data/data.json"):

        # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device="CPU")
        plugin.add_cpu_extension(cpu_lib)

        # Read detector IR
        detector_bin = os.path.splitext(detector_xml)[0] + ".bin"
        log.info("Loading detector network files:\n\t{}\n\t{}".format(detector_xml, detector_bin))
        detector_net = IENetwork.from_ir(model=detector_xml, weights=detector_bin)
        
        

        # Read landmarks IR
        landmarks_bin = os.path.splitext(landmarks_xml)[0] + ".bin"
        log.info("Loading landmarks network files:\n\t{}\n\t{}".format(landmarks_xml, landmarks_bin))
        landmarks_net = IENetwork.from_ir(model=landmarks_xml, weights=landmarks_bin)

        # Read features IR
        features_bin = os.path.splitext(features_xml)[0] + ".bin"
        log.info("Loading features network files:\n\t{}\n\t{}".format(features_xml, features_bin))
        features_net = IENetwork.from_ir(model=features_xml, weights=features_bin)

        # Read age-gender IR
        age_gender_bin = os.path.splitext(age_gender_xml)[0] + ".bin"
        log.info("Loading age-gender network files:\n\t{}\n\t{}".format(age_gender_xml, age_gender_bin))
        age_gender_net = IENetwork.from_ir(model=age_gender_xml, weights=age_gender_bin)

        log.info("Preparing blobs")
        self.d_in = next(iter(detector_net.inputs))
        self.d_out = next(iter(detector_net.outputs))
        detector_net.batch_size = 1

        self.l_in = next(iter(landmarks_net.inputs))
        self.l_out = next(iter(landmarks_net.outputs))
        landmarks_net.batch_size = 1

        self.f_in = next(iter(features_net.inputs))
        self.f_out = next(iter(features_net.outputs))
        features_net.batch_size = 1

        self.a_in = next(iter(age_gender_net.inputs))
        self.a_age_out = "age_conv3"
        self.a_gender_out = "prob"
        age_gender_net.batch_size = 1

        # Read and pre-process input images
        self.d_n, self.d_c, self.d_h, self.d_w = detector_net.inputs[self.d_in]
        self.d_images = np.ndarray(shape=(self.d_n, self.d_c, self.d_h, self.d_w))

        self.l_n, self.l_c, self.l_h, self.l_w = landmarks_net.inputs[self.l_in]
        self.l_images = np.ndarray(shape=(self.l_n, self.l_c, self.l_h, self.l_w))

        self.f_n, self.f_c, self.f_h, self.f_w = features_net.inputs[self.f_in]
        self.f_images = np.ndarray(shape=(self.f_n, self.f_c, self.f_h, self.f_w))

        self.a_n, self.a_c, self.a_h, self.a_w = age_gender_net.inputs[self.a_in]
        self.a_images = np.ndarray(shape=(self.a_n, self.a_c, self.a_h, self.a_w))

        # Loading models to the plugin
        log.info("Loading models to the plugin")
        self.d_exec_net = plugin.load(network=detector_net)
        # print("out", inspect.getmembers(self.d_exec_net, predicate=inspect.ismethod))
        # print("doc", self.d_exec_net.__doc__)
        # print("repr", self.d_exec_net.__repr__)
        # print("class", self.d_exec_net.__class__)
        
        self.l_exec_net = plugin.load(network=landmarks_net)
        self.f_exec_net = plugin.load(network=features_net)
        self.a_exec_net = plugin.load(network=age_gender_net)

        self.detection_threshold = detection_threshold
        self.face_aligner = FaceAligner(face_width=self.f_w, face_height=self.f_h)

        if train:
            self.data = []

            for person_dir in os.listdir(face_samples):
                print(person_dir)

                for img_file in os.listdir(os.path.join(face_samples, person_dir)):
                    print(img_file)
                    image = cv2.imread(os.path.join(face_samples, person_dir, img_file))
                    faces = self.get_aligned_faces(image)
                    if len(faces) > 0:
                        face = faces[0][-1]
                        self.data.append((self.face_to_vector(face).tolist(), person_dir))
            with open(data_path, mode="w") as data_file:
                json.dump(self.data, data_file)
        else:
            with open(data_path, mode="r") as data_file:
                self.data = json.load(data_file)

    def get_aligned_faces(self, frame):
        height, width = frame.shape[:-1]
        if (height, width) != (self.d_h, self.d_w):
            d_frame = cv2.resize(frame, (self.d_w, self.d_h))
        else:
            d_frame = frame
        
        # Change data layout from HWC to CHW
        self.d_images[0] = d_frame.transpose((2, 0, 1))
        import numpy as np
        a = np.array([])
        print("out", inspect.getmembers(np, predicate=inspect.ismethod))
        # t0 = time()
        d_res = self.d_exec_net.infer(inputs={self.d_in: self.d_images})[self.d_out][0][0]
        # print("d_res", d_res)
        # total_time += time() - t0
        
        aligned_faces = []
        for number, label, confidence, left, top, right, bottom in d_res:
            left = max(0, int(left * width))
            right = min(int(right * width), width - 1)
            top = max(0, int(top * height))
            bottom = min(int(bottom * height), height - 1)
            if confidence >= self.detection_threshold:
                face = cv2.resize(frame[top:bottom, left:right], (self.l_w, self.l_h))
                self.l_images[0] = face.transpose((2, 0, 1))
                l_res = np.squeeze(self.l_exec_net.infer(inputs={self.l_in: self.l_images})[self.l_out])
                for i in range(10):
                    if i % 2 == 0:
                        l_res[i] = left + (right - left) * l_res[i]
                    else:
                        l_res[i] = top + (bottom - top) * l_res[i]
                aligned_face = self.face_aligner.align(frame, l_res)
                aligned_faces.append((left, right, top, bottom, aligned_face))

        return sorted(aligned_faces, key=lambda x: x[1] - x[0], reverse=True)

    def face_to_vector(self, face):
        self.f_images[0] = face.transpose((2, 0, 1))
        f_res = np.squeeze(self.f_exec_net.infer(inputs={self.f_in: self.f_images})[self.f_out])
        return f_res

    def find_closest(self, face):
        min_name = None
        min_dist = None
        vector = self.face_to_vector(face)
        for sample_vector, sample_name in self.data:
            dist = spatial.distance.cosine(vector, np.asarray(sample_vector))
            if min_name is None or dist < min_dist:
                min_dist = dist
                min_name = sample_name

        return min_name, 1.0 - min_dist

    def get_age_gender(self, aligned_face):
        a_face = cv2.resize(aligned_face, (self.a_w, self.a_h))

        # Change data layout from HWC to CHW
        self.a_images[0] = a_face.transpose((2, 0, 1))

        a_res = self.a_exec_net.infer(inputs={self.a_in: self.a_images})

        age = a_res[self.a_age_out][0][0][0][0] * 100
        gender = "female" if a_res[self.a_gender_out][0][0][0][0] > a_res[self.a_gender_out][0][1][0][0] else "male"
        return age, gender

    def detect_and_identify(self, frame):
        aligned_faces = self.get_aligned_faces(frame)
        # print("aligned_faces", aligned_faces)
        result = []
        for left, right, top, bottom, aligned_face in aligned_faces:
            min_name, min_dist = self.find_closest(aligned_face)
            age, gender = self.get_age_gender(aligned_face)
            result.append((left, right, top, bottom, min_name, min_dist, aligned_face, age, gender))

        return result