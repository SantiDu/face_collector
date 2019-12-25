import numpy as np
import cv2

def get_cammtx(im):
    h, w, c = im.shape
    f = w # column size = x axis length (focal length)
    u0, v0 = w / 2, h / 2 # center of image plane
    return np.array(
        [[f, 0, u0],
        [0, f, v0],
        [0, 0, 1]], dtype = np.double
        )
    
    
# Assuming no lens distortion
dist_coeffs = np.zeros((4,1)) 

object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33 left brow left corner
                         [1.330353, 7.122144, 6.903745],  #29 left brow right corner
                         [-1.330353, 7.122144, 6.903745], #34 right brow left corner
                         [-6.825897, 6.760612, 4.402142], #38 right brow right corner
                         [5.311432, 5.485328, 3.987654],  #13 left eye left corner
                         [1.789930, 5.393625, 4.413414],  #17 left eye right corner
                         [-1.789930, 5.393625, 4.413414], #25 right eye left corner
                         [-5.311432, 5.485328, 3.987654], #21 right eye right corner
                         [2.005628, 1.409845, 6.165652],  #55 nose left corner
                         [-2.005628, 1.409845, 6.165652], #49 nose right corner
                         [2.774015, -2.080775, 5.048531], #43 mouth left corner
                         [-2.774015, -2.080775, 5.048531],#39 mouth right corner
                         [0.000000, -3.116408, 6.097667], #45 mouth central bottom corner
                         [0.000000, -7.415691, 4.070434]])#6 chin corner


def get_head_pose(shape, cam_matrix, dist_coeffs=dist_coeffs):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angle
