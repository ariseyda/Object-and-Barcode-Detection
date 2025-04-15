# 🔍 Real-Time Object and Barcode Detection

This project combines **object detection** and **barcode/QR code scanning** in real time using a webcam. It uses a pre-trained MobileNet SSD model (trained on the COCO dataset) for object detection, and the `pyzbar` library for barcode decoding.

📷 Detect objects like persons, cars, and more  
📦 Identify and decode barcodes and QR codes on the fly  
🖥️ Live feed with bounding boxes, class labels, confidence scores, and barcode data  

---

## 🎯 Features

- Real-time object detection using **SSD MobileNet v3 (COCO dataset)**
- Real-time barcode and QR code scanning using **pyzbar**
- Displays class names, bounding boxes, and confidence levels
- Highlights and decodes barcodes with polygon overlays
- Collects and prints **unique object classes** and **unique barcodes**

---

## 🛠 Technologies Used

- **Python 3**
- **OpenCV** – Image processing and real-time video capture
- **TensorFlow Object Detection** – Pre-trained SSD MobileNet v3 model
- **pyzbar** – Barcode and QR code detection
- **NumPy, Pandas** – Data processing and optional analysis

---

## 📌 Requirements
A webcam or USB camera

Python 3.6+

Pre-trained model files:

frozen_inference_graph.pb

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

coco.names

Model files can be downloaded from the official TensorFlow Object Detection API release:

Download SSD MobileNet v3 (COCO)

---

## 📸 Output Example
Objects are shown with green bounding boxes and labels

Barcodes/QR codes are shown with pink polygon outlines and decoded text
