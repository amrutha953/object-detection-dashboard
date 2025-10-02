# object-detection-dashboard
🧠 Real-Time Object Detection using YOLOv5 and Streamlit

This project is a Streamlit-based web application that performs real-time object detection using YOLOv5
. It supports detection from webcam, uploaded images, and videos.

🔍 Features

Live object detection from webcam.

Upload image or video files for detection.

Adjustable confidence and IoU thresholds.

Real-time visualizations with bounding boxes and labels.

Powered by YOLOv5, OpenCV, and Streamlit.

🖼️ Demo

<!-- Optional: Add a demo gif or image -->

🚀 Getting Started
1. Clone the repository
git clone https://github.com/your-username/yolov5-streamlit-detection.git
cd yolov5-streamlit-detection

2. Install dependencies

We recommend using a virtual environment:

pip install -r requirements.txt


Or manually:

pip install streamlit opencv-python-headless numpy ultralytics


⚠️ opencv-python-headless is used to avoid GUI issues with Streamlit. Replace with opencv-python if running outside a server/headless environment.

3. Download YOLOv5 model weights

The script uses the small YOLOv5 model by default (yolov5s.pt). It will automatically download on first run, but you can also manually download from Ultralytics YOLOv5 Releases
.

▶️ Running the App
streamlit run app.py
