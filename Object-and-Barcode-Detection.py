import cv2
from pyzbar.pyzbar import decode
import numpy as np
import pandas as pd

barcodes=[]
objects=[]


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

img=cv2.imread("barcode.jpg")

#img=cv2.imread("car.png")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
#cap.set(10, 70)
#cap.set(3,640)#width
#cap.set(4,480)#height


thres = 0.45  # Threshold to detect object

classNames=[]
classFile="coco.names"
with open(classFile,"rt") as f:
    classNames=f.read().rstrip("\n").split("\n")


    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    cap = cv2.VideoCapture(0)

    while True:

        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)


        if len(classIds) != 0:

            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                objects.append(classNames[classId - 1].upper())

                for barcode in decode(img):
                    # print(barcode.data)  # tells us whether is barcode or qrcode
                    myData = barcode.data.decode("utf-8")  # convert to string

                    print(myData)
                    barcodes.append(myData)

                    pts = np.array([barcode.polygon], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, (255, 0, 255), 5)
                    pts2 = barcode.rect
                    cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 255), 2)


        cv2.imshow("Result", img)

        #cv2.waitKey(1)
        key = cv2.waitKey(1)
        if key == ord('q'):  # key==27
            cap.release()

            unique_barcodes=len(set(barcodes))
            unique_objects=len(set(objects))
            print("Number of detected barcodes: %d" % unique_barcodes)
            #print("Number of detected objects: %d" % unique_objects)
            
            break



