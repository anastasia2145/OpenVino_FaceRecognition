import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import cv2
import matplotlib.pylab as plt

from PRNet.api import PRN
from PRNet.utils.write import write_obj_with_colors


def show(image):
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt.show()
    
# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = True) 

save_folder = 'mytest_result'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

image_path = 'el_safadi_33.jpg'
name = 'el_safadi_33'
image = imread(image_path)
# image = resize(image, (256,256))
n, m, _ = image.shape
# image_info = np.array([left, right, top, bottom])
pos = prn.process(image, image_info=np.array([0, m, 0, n]))
# pos = prn.process(image)
print(pos)
cv2.imshow('image', image)
cv2.waitKey(0)

print("pos", pos.shape)
print("image", image.shape)
kpt = prn.get_landmarks(pos)

vertices = prn.get_vertices(pos)
print("vertices", vertices.shape)
print(vertices)
colors = prn.get_colors(image, vertices)
print("colors", colors.shape)
np.savetxt(os.path.join(save_folder, name + '.txt'), kpt) 
write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})