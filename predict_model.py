# Import library
import torch
import cv2
import numpy as np
from random import randrange
import os

# Plot rectangle and text into original image
def plot_label(label, image_size, image):
    class_arr = []
    c = 0

    Rx = image_size[1]/640
    Ry = image_size[0]/640

    for i in label:
        print(i)
        x_center = int(i[0])
        y_center = int(i[1])
        w = int(i[2])
        h = int(i[3])

        x1 = int((x_center - w//2)*Rx)
        x2 = int((x_center + w//2)*Rx)
        y1 = int((y_center - h//2)*Ry)
        y2 = int((y_center + h//2)*Ry)

        for j in range(len(classes)):
            class_arr.append(i[j+5])
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_thickness = 1
        font_size = h/450

        if font_size < 0.5:
            font_size = 0.5

        if len(colors) <= c:
            colors.append([randrange(255),randrange(255),randrange(255)])

        color = (colors[c][0], colors[c][1], colors[c][2])

        idx = np.argmax(class_arr)
            
        objects = classes[idx]

        (w, h), _ = cv2.getTextSize(objects + " - " + str("{:.2f}".format(class_arr[idx])), font, font_size, font_thickness)

        cv2.rectangle(image, (x1, y1), (x1+w, y1-h), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        cv2.putText(image, objects + " - " + str("{:.2f}".format(class_arr[idx])), (x1,y1), font, font_size,(255,255,255), font_thickness)

        class_arr.clear()
        c += 1

    return image

# Find the intersected label and exclude the one with lower accuracy
def find_unintersect_label(label):
    detected = []

    for i in label[0]:
        if i[4] >= 0.3:
            if len(detected) == 0:
                detected.append(i)
            else:
                ax_center = int(i[0])
                ay_center = int(i[1])
                aw = int(i[2])
                ah = int(i[3])

                ax1 = ax_center - aw//2
                ax2 = ax_center + aw//2
                ay1 = ay_center - ah//2
                ay2 = ay_center + ah//2

                similar = False

                for j in range(len(detected)):
                    bx_center = int(detected[j][0])
                    by_center = int(detected[j][1])
                    bw = int(detected[j][2])
                    bh = int(detected[j][3])

                    bx1 = bx_center - bw//2
                    bx2 = bx_center + bw//2
                    by1 = by_center - bh//2
                    by2 = by_center + bh//2

                    x1 = max(min(ax1, ax2), min(bx1, bx2))
                    y1 = max(min(ay1, ay2), min(by1, by2))
                    x2 = min(max(ax1, ax2), max(bx1, bx2))
                    y2 = min(max(ay1, ay2), max(by1, by2))
                    
                    if x1<x2 and y1<y2:
                        inter_area = (x2-x1)*(y2-y1)
                        diff_area = (aw*ah - inter_area) + (bw*bh - inter_area)

                        if inter_area > diff_area:
                            similar = True
                            if detected[j][4] < i[4]:
                                detected[j] = i
                                break

                if not similar:                    
                    detected.append(i)

    return detected

# Predict label using yolov5 custom model
def predict_output(img):
    img_size = img.shape
    img_ori = img
    img = cv2.resize(img, (640, 640), interpolation = cv2.INTER_AREA)

    img = img[:, :, ::-1]
    img = img.reshape(1,640,640,3)
    img = torch.from_numpy(np.flip(img,axis=0).copy())
    img = img.permute(0, 3, 1, 2)/255

    results = model(img)

    results = results.detach().numpy()

    detected = find_unintersect_label(label=results)

    img_ori = plot_label(label=detected, image=img_ori, image_size=img_size)

    return img_ori

# Read input between realtime or non-realtime
def read_input(path):
    if (path.split('.')[1] == "mp4"):
        # Capture frame from video input
        video_path = os.path.join("video/cars.mp4")
        cap = cv2.VideoCapture(video_path)
        # Process frame as an input 
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image_pred = predict_output(frame)
            cv2.imshow("image", image_pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # Read image locally
        input_image = cv2.imread(path)

        image_pred = predict_output(input_image)
        cv2.imshow("image", image_pred)
        cv2.waitKey(0)

# Define classes name
classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

# List of colors
colors = []

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolo_model/model_cars.torchscript')

# Image path for local input
path = "video/cars.mp4"
# path = "image/motorcycle.jpg"

# Read input
read_input(path=path)