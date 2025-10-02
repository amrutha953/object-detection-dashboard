import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

st.title("Real-Time Object Detection using YOLOv5")
st.write("Detect objects live using your webcam or upload images/videos. Powered by YOLOv5, OpenCV, and Streamlit.")

# Load YOLOv5 model
model = YOLO("yolov5s.pt")  # Use 'yolov5s.pt' for speed

# Sidebar controls
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)

mode = st.radio("Select Mode", ("Webcam", "Image", "Video"))

def infer_and_draw(image):
    results = model(image)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {box.conf[0]:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image

if mode == "Image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        out_img = infer_and_draw(image)
        st.image(out_img, channels="BGR")

elif mode == "Video":
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = infer_and_draw(frame)
            stframe.image(out_frame, channels="BGR")
        cap.release()

else:  # Webcam
    run_webcam = st.button("Start Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = infer_and_draw(frame)
            stframe.image(out_frame, channels="BGR")
            if st.button("Stop"):
                break
        cap.release()


