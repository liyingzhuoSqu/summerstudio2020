# Import required modules
import cv2 as cv
import argparse
import label_map_util
import visualization_utils as vis_util
from visualization_utils import *
import numpy as np
import time
import os
import random
import shutil
import tensorflow as tf
from test import delete

delete()

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()
# facial Recognition
faceProto = "Gender_model/opencv_face_detector.pbtxt"
faceModel = "Gender_model/opencv_face_detector_uint8.pb"
# load in Gender
genderProto = "Gender_model/gender_deploy.prototxt"
genderModel = "Gender_model/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load network
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


def face():
    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                   max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            label = "{}".format(gender)
            cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                       cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
            print(gender)
            # print(age)
            if gender is 'Male':
                print('boy')
                return 1
            elif gender is 'Female':
                print('girl')
                return 2
        if gender == 1 or gender == 2:
            cap.release()
            cv.destroyAllWindows()
            break


def clothes():
    MODEL_NAME = 'clothing_model'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'clothing_Model', 'label_map.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 10

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv.VideoCapture(0)
    ret = video.set(3, 1280)
    ret = video.set(4, 720)

    while (True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        for index, value in enumerate(classes[0]):
            if scores[0, index] > 0.5:
                Cloth = category_index.get(value)['name']
                # print(Cloth)
                if Cloth == 'dresses':
                    return 1
                if Cloth == 'handbags':
                    return 2
                if Cloth == 'jacket':
                    return 3
                if Cloth == 'jeans':
                    return 4
                if Cloth == 'shirts':
                    return 5
                if Cloth == 'shorts':
                    return 6
                if Cloth == 'suits':
                    return 7
                if Cloth == 'tops':
                    return 8
                if Cloth == 'trousers':
                    return 9
                if Cloth == 'tees':
                    return 10
                if 0 < Cloth < 10:
                    video.release()
                    cv.destroyAllWindows()
                    break

        # All the results have been drawn on the frame, so it's time to display it.
        cv.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            video.release()
            cv.destroyAllWindows()
            break


def ethnicity():
    MODEL_NAME = 'ethnicity_model'
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'ethnicity_model', 'labelmap.pbtxt')
    # Number of classes the object detector can identify
    NUM_CLASSES = 10
    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Initialize webcam feed
    video = cv.VideoCapture(0)
    ret = video.set(3, 1280)
    ret = video.set(4, 720)

    while (True):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        for index, value in enumerate(classes[0]):
            if scores[0, index] > 0.5:
                eth = category_index.get(value)['name']
                if eth == "asian_male":
                    return 1
                if eth == "asian_female":
                    return 2
                if eth == "Indian_Male":
                    return 3
                if eth == "indian_female":
                    return 4
                if eth == "white_male":
                    return 5
                if eth == "white_female":
                    return 6
                if eth == "middle_eastern_male":
                    return 7
                if eth == "middle_eastern_female":
                    return 8
                if eth == "black_male":
                    return 9
                if eth == "black_female":
                    return 10
                if 0<eth<10:
                    video.release()
                    cv.destroyAllWindows()
                    break
        # All the results have been drawn on the frame, so it's time to display it.
        # cv.imshow('Object detector', frame)
        # Press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            video.release()
            cv.destroyAllWindows()
            break


output_gender = face()
print(output_gender)
Ad_folder = 'slideshow/Ad_images/'
if output_gender == 1:
    male = random.choice(os.listdir('slideshow/male/'))
    shutil.copy('slideshow/male/' + male, Ad_folder + male)
    print('male')
if output_gender == 2:
    female = random.choice(os.listdir('slideshow/female/'))
    shutil.copy('slideshow/female/' + female, Ad_folder + female)
    print('female')
# time.sleep(50 / 1000)  # delay(50)

output_ethnicity = ethnicity()
print(output_ethnicity)
if output_ethnicity == 1 or output_ethnicity == 2:
    asian = random.choice(os.listdir('slideshow/asian/'))
    shutil.copy('slideshow/asian/' + asian, Ad_folder + asian)
    print('asian')
if output_ethnicity == 3 or output_ethnicity == 4:
    indian = random.choice(os.listdir('slideshow/indian/'))
    shutil.copy('slideshow/indian/' + indian, Ad_folder + indian)
    print('indian')
if output_ethnicity == 5 or output_ethnicity == 6:
    white = random.choice(os.listdir('slideshow/white/'))
    shutil.copy('slideshow/white/' + white, Ad_folder + white)
    print('white')
if output_ethnicity == 7 or output_ethnicity == 8:
    middle_eastern = random.choice(os.listdir('slideshow/middle_eastern/'))
    shutil.copy('slideshow/middle_eastern/' + middle_eastern, Ad_folder + middle_eastern)
    print('middle eastern')
if output_ethnicity == 9 or output_ethnicity == 10:
    black = random.choice(os.listdir('slideshow/black/'))
    shutil.copy('slideshow/black/' + black, Ad_folder + black)
    print('black')

# time.sleep(50 / 1000)  # delay(50)
output_clothes = clothes()
print(output_clothes)
if output_clothes == 1:
    dresses = random.choice(os.listdir('slideshow/dresses/'))
    shutil.copy('slideshow/dresses/' + dresses, Ad_folder + dresses)
    print('dresses')
if output_clothes == 2:
    handbag = random.choice(os.listdir('slideshow/handbag/'))
    shutil.copy('slideshow/handbag/' + handbag, Ad_folder + handbag)
    print('handbag')
if output_clothes == 3:
    jacket = random.choice(os.listdir('slideshow/jacket/'))
    shutil.copy('slideshow/jacket/' + jacket, Ad_folder + jacket)
    print('jacket')
if output_clothes == 4:
    jeans = random.choice(os.listdir('slideshow/jeans/'))
    shutil.copy('slideshow/jeans/' + jeans, Ad_folder + jeans)
    print('jeans')
if output_clothes == 5:
    shirts = random.choice(os.listdir('slideshow/shirts/'))
    shutil.copy('slideshow/shirts/' + shirts, Ad_folder + shirts)
    print('shirts')
if output_clothes == 6:
    shorts = random.choice(os.listdir('slideshow/shorts/'))
    shutil.copy('slideshow/shorts/' + shorts, Ad_folder + shorts)
    print('shorts')
if output_clothes == 7:
    suits = random.choice(os.listdir('slideshow/suits/'))
    shutil.copy('slideshow/suits/' + suits, Ad_folder + suits)
    print('suits')
if output_clothes == 8:
    tops = random.choice(os.listdir('slideshow/tops/'))
    shutil.copy('slideshow/tops/' + tops, Ad_folder + tops)
    print('tops')
if output_clothes == 9:
    trousers = random.choice(os.listdir('slideshow/trousers/'))
    shutil.copy('slideshow/black/' + trousers, Ad_folder + trousers)
    print('trousers')
if output_clothes == 10:
    tees = random.choice(os.listdir('slideshow/tees/'))
    shutil.copy('slideshow/tees/' + tees, Ad_folder + tees)
    print('tees')
# time.sleep(50 / 1000)  # delay(50)

from itertools import cycle
import tkinter as tk
import os
# foreign library, need to installed
from PIL.ImageTk import PhotoImage


def dynad():
    path = 'slideshow/Ad_images'
    N = int(len(path))
    i = 0
    images = []
    for files in os.listdir(path):
        if files.endswith('.jpg'):
            img = 'slideshow/Ad_images/' + files
            img1 = [img]
            image = images
            images = image + img1
            i += 1
            if i == N:
                break

    # print (images)

    class Imagewindow(tk.Tk):
      def __init__(self):
          tk.Tk.__init__(self)
          self.photos = cycle(
              PhotoImage(file=image) for image in images
          )
          # self.printButton =Imagewindow(frame, image=image[0], command=self.nextPizza)
          self.title("Button GUI")
          self.displayCanvas = tk.Label(self)
          self.displayCanvas.pack()




      def slideShow(self):
          img = next(self.photos)
          self.displayCanvas.config(image=img)
          # self.title('Targetingly')
          self.after(2500, self.slideShow)  # 0.05 seconds

      def run(self):
          self.mainloop()

    root = Imagewindow()
    width = root.winfo_screenwidth()
    height = root.winfo_screenwidth()
    root.overrideredirect(True)
    root.geometry('%dx%d' % (width * 1, height * 1))
    root.slideShow()
    root.run()

dynad()

# if cv.waitKey(1) == ord('q'):
#     video.release()
#     cv.destroyAllWindows()
# delete()


# a = 0
# if a<30:
#     print(a)
#     time.sleep(1)
#     slideShow.destroy()
#     a+=1
# test()
