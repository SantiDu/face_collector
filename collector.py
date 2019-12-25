import cv2
import numpy as np
import dlib
import os
# from ultra_face_opencvdnn_inference import inference
from headpose import get_cammtx, get_head_pose
from imutils import face_utils
from ypr_angle import ypr

face_detector = dlib.get_frontal_face_detector()
lm_detector = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
fa = face_utils.facealigner.FaceAligner(lm_detector, desiredLeftEye=(0.3, 0.3))

def angles2lines(angles, radius, offset=5, length=10):
    lines = []
    for angle in angles:
        cos = np.cos(angle * np.pi / 180)
        sin = np.sin(angle * np.pi / 180)
        p1_x = np.uint(u0 + (radius + offset) * cos)
        p1_y = np.uint(v0 + (radius + offset) * sin)
        p2_x = np.uint(u0 + (radius + offset + length) * cos)
        p2_y = np.uint(v0 + (radius + offset + length) * sin)
        lines.append((p1_x, p1_y, p2_x, p2_y))
    return lines

def creat_dir(path):
    if os.path.isdir(path):
        print ("the directory %s exits" % path)
    else:
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

path = './chopped_images'
creat_dir(path)
cap = cv2.VideoCapture(0)
count = 0
capturing = 0
drawing = 0
color = (255, 255, 255)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    mask = np.zeros((h, w), np.uint8)
    u0, v0 = w // 2, h // 2
    radius =  min(h,w)//3
    cv2.circle(mask, (u0, v0), radius, color, thickness=-1)
    masked_data = cv2.bitwise_and(frame, frame, mask=mask)
    
    radius_diff = 10
    for data in ypr:
        if data[2] == 1:
            color = (0, 255, 0)
            radius_diff = 20
        angles = range(data[1][0], data[1][1], 2)
        lines = angles2lines(angles, radius, length=radius_diff)
        for points in lines:
            p1_x, p1_y, p2_x, p2_y = points
            cv2.line(masked_data, (p1_x, p1_y), (p2_x, p2_y), color, thickness=1)
        color = (255, 255, 255)
        radius_diff = 10

    # Detect faces
    # rgb_img = cv2.cvtColor(masked_data, cv2.COLOR_BGR2RGB)
    # boxes, _, _ = inference(rgb_img)
    gray = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    # Detect landmarks
    # if boxes.shape[0]:
    #     box = boxes[0, :]
    #     x1, y1, x2, y2 = box
    if any(faces):
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(masked_data, (x1, y1), (x2, y2), (255, 255, 0), 3)
        landmarks = lm_detector(gray, face)
        # landmarks = lm_detector(rgb_img, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
        shape = face_utils.shape_to_np(landmarks)
        euler_angle = get_head_pose(shape, get_cammtx(frame))
        pitch = euler_angle[0, 0]
        yaw = euler_angle[1, 0]
        roll = euler_angle[2, 0]
        cv2.putText(masked_data, "X: " + "{:7.2f}".format(pitch), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        cv2.putText(masked_data, "Y: " + "{:7.2f}".format(yaw), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        cv2.putText(masked_data, "Z: " + "{:7.2f}".format(roll), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        # for x, y in landmarks[0][0]:
        for x, y in shape:
            cv2.circle(masked_data, (x, y), 1, (255, 0, 0), -1)
        if capturing == 0:
            for pr in ypr:
                if pr[2] == 0: 
                    if pr[0][0] - 3 < pitch < pr[0][0] + 3 and pr[0][1] - 5 < yaw < pr[0][1] + 5 and -15 < roll < 15:
                        capturing = 1
                        drawing = 1
                        index = 0
                        # align and resize
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                        # aligned_face = cv2.resize(aligned_face, (112,112))
                        # save aligned face
                        cv2.imwrite(f"{path}/aligned_face{pr[3]}_yaw{round(yaw,1)}_pitch{round(pitch,1)}_roll{round(roll,1)}.jpg", aligned_face)
                        break
    if drawing == 1:
        angles = range(pr[1][0], pr[1][1], 2)
        radius_diff = 20
        lines = angles2lines(angles, radius, length=radius_diff)
        freq = 3000
        if count % freq == 0:
            for points in lines[:index]:
                p1_x, p1_y, p2_x, p2_y = points
                cv2.line(masked_data, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), thickness=1)
            index += 1
            if index == len(lines):
                pr[2] = 1
                capturing = 0
                drawing = 0
                count = 0
        while count < freq * len(lines):
            count += 1
    # Display the resulting frame
    cv2.imshow('frame',masked_data)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
