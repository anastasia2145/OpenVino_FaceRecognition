import numpy as np
import cv2


class FaceAligner:
    def __init__(self, left_eye=(0.31, 0.46), left_lip=(0.35, 0.82),
                 face_width=128, face_height=None):
        self.left_eye = left_eye
        self.left_lip = left_lip
        self.face_width = face_width
        self.face_height = face_height
        if self.face_height is None:
            self.face_height = self.face_width

        self.desired_eyes_dist = (1.0 - 2 * self.left_eye[0]) * self.face_width
        self.desired_eye_lip_dist = (self.left_lip[1] - self.left_eye[1]) * self.face_height

    def align(self, image, l_res):
        # compute the angle between the eye centroids
        d_x = l_res[2] - l_res[0]
        d_y = l_res[3] - l_res[1]
        angle = np.degrees(np.arctan2(d_y, d_x))

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist_x = np.sqrt((d_x ** 2) + (d_y ** 2))
        scale_x = self.desired_eyes_dist / dist_x

        # calculate scale_y based on the same rule
        dist_y = (l_res[7] - l_res[1]) * np.cos(np.arctan2(d_y, d_x))
        scale_y = self.desired_eye_lip_dist / dist_y

        scale = min(scale_x, scale_y)

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((l_res[0] + l_res[2]) / 2, (l_res[1] + l_res[3]) / 2)

        # grab the rotation matrix for rotating and scaling the face
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = self.face_width * 0.5
        t_y = self.face_height * self.left_eye[1]
        matrix[0, 2] += (t_x - eyes_center[0])
        matrix[1, 2] += (t_y - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.face_width, self.face_height)
        output = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output
