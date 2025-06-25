# **Face Recognition System**

A robust facial recognition application built with Python, utilizing custom trained deep learning models for face detection and recognition.

## **Features**

* Real-time face detection and recognition with **configurable camera sources (Built-in, ESP32-CAM)**.  
* **Multiple Face Detection Model Support:** Easily switch between **YuNet, Haar Cascade, and a Custom CNN model**.  
* Custom-trained models for enhanced accuracy.  
* User registration and authentication system.  
* Access control management.  
* **Enhanced GUI interface** built with CustomTkinter, featuring **tabbed navigation for Main Controls, Settings, and Person Management**.  
* Support for IP camera integration (e.g., ESP32-CAM).  
* Queue-based embedding averaging for better recognition robustness (num\_samples configurable).  
* Thread-safe database operations for stable performance.  
* **Real-time display of recognition threshold and last match distance.**  
* **In-app logging to GUI textbox.**

## **Technical Stack**

* **Language:** Python  
* **GUI Framework:** CustomTkinter  
* **Deep Learning:** PyTorch  
* **Computer Vision:** OpenCV  
* **Database:** SQLite  
* **Camera Interface:** Built-in Webcams, ESP32-CAM (via HTTP stream)

## **Models**

### **Face Detection**

The system now offers **multiple configurable face detection models** for flexibility:

* **YuNet Detector:**  
  * Model: face\_detection\_yunet\_2023mar.onnx  
  * Integrated via cv2.FaceDetectorYN for robust and efficient real-time face detection, especially suitable for varying image qualities including those from ESP32-CAM.  
  * Input size: (320, 320\) pixels.  
  * Configurable confidence and NMS thresholds.  
* **Haar Cascade Detector:**  
  * Standard OpenCV Haar Cascade Classifier (haarcascade\_frontalface\_default.xml).  
  * A classic and lightweight option for face detection.  
* **Custom CNN Detector:**  
  * Model: bbox\_v5\_randomly\_augmented\_epoch\_3.pth (or similar version from models/bbox\_models/v5/)  
  * Custom trained model on CelebA dataset.  
  * Model architecture: CNN with multiple convolutional and fully connected layers.  
  * Input size: 224x224 pixels.  
  * Output: 4 coordinates (x, y, width, height).  
  * Training dataset: CelebA with 200k+ images ([Hugging Face Dataset](https://huggingface.co/datasets/hfaus/CelebA_bbox_and_facepoints)).  
  * Training notebook: models/bbox\_models/v5/facial-detection-celebA-bbox-optimized.ipynb ([Kaggle Training Notebook](https://www.kaggle.com/code/ahmedkamal75/facial-detection-celeba-bbox-optimized)).  
  * Models:  
    * Without Augmentations: [Kaggle Model](https://www.kaggle.com/models/ahmedkamal75/bbox_model_v5_epoch_8/)  
    * With Augmentations: [Kaggle Model](https://www.kaggle.com/models/ahmedkamal75/bbox_v5_augmented_epoch_50)

### **Face Recognition**

* ArcFace-based architecture with residual connections.  
* Embedding dimension: 512\.  
* Loss function: ArcFace loss with dynamic scale and margin parameters.  
* Preprocessing: Face alignment using detected landmarks.  
* Training dataset: [VGGFace2-HQ Cropped](https://www.kaggle.com/datasets/zenbot99/vggface2-hq-cropped) (Kaggle).  
* Testing dataset: [PINS Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) (Kaggle).  
* Training notebook: models/resarksgd/arkface-residual-connections-part-2.ipynb ([Kaggle Training Notebook](https://www.kaggle.com/code/ahmedkamal75/arkface-residual-connections-part-2)).  
* Models:  
  * Augmented 94% accuracy: [Kaggle Notebook](https://www.kaggle.com/code/ahmedkamal75/arkface-residual-connections-part-2/notebook?scriptVersionId=217035765)

## **Requirements**

* Python 3.8+  
* CUDA capable GPU (recommended for faster processing)  
* ESP32-CAM (optional, for IP camera usage)  
* Minimum 4GB RAM  
* Storage: \~500MB for models

## **Project Structure**
```
Graduation\_Project/  
├── app\_V2.py              \# Main application with enhanced features and GUI  
├── app\_V1.py              \# Previous main application version (renamed from app.py)  
├── embeddings.py          \# Face recognition model implementation  
├── bounding\_box.py        \# Custom CNN face detection model implementation (and Haar Cascade integration)  
├── bounding\_box\_yunet.py  \# YuNet face detector implementation  
└── models/  
    ├── bbox\_models/       \# Face detection models  
        ├── v5/            \# Custom trained models (e.g., bbox\_v5\_randomly\_augmented\_epoch\_3.pth)  
        └── YuNet/         \# YuNet face detector models (e.g., face\_detection\_yunet\_2023mar.onnx)  
    └── resarksgd/         \# Face recognition models (e.g., resarksgdaug94.pth, resarksgd95.pth)
```
## **Installation**

1. Clone the repository:  
   git clone https://github.com/AhmedKamal75/Graduation\_Project.git  
   cd Graduation\_Project

2. Create a virtual environment and activate it:  
   python3 \-m venv .venv      \# Using .venv is a common convention  
   .venv\\Scripts\\activate     \# Windows

3. Install the required dependencies:  
   pip install \-r requirements.txt

4. Download the necessary pre-trained models:  
   * **Face Recognition Model (e.g.,** resarksgdaug94.pth **or** resarksgd95.pth **):**  
     * Download from: [Kaggle Model \- 94% accuracy](https://www.kaggle.com/code/ahmedkamal75/arkface-residual-connections-part-2/notebook?scriptVersionId=217035765) , you can download it from the output section.
     * Place the downloaded .pth file in the models/resarksgd/ directory.  
   * **YuNet Face Detector Model** (face\_detection\_yunet\_2023mar.onnx **):**  
     * Download if not already present (often included with OpenCV or specific downloads). Ensure it is in models/bbox\_models/YuNet/.  
   * **Custom CNN Face Detector Model (e.g.,** bbox\_v5\_randomly\_augmented\_epoch\_3.pth **):**  
     * Download if you intend to use this model. Ensure it is in models/bbox\_models/v5/.  
5. No manual path updates are needed in app\_V2.py for model paths, as they are specified at the top of the if \_\_name\_\_ \== "\_\_main\_\_": block. However, ensure the files exist at the specified paths.

## **Usage Examples**

1. Run the main application:  
   python app\_V2.py

2. The GUI will launch with a video feed and control panels.

### **Registration Process**

1. Navigate to the "Main Controls" tab.  
2. Click "Register Person".  
3. Enter the person's name and set access permissions (Allowed Access checkbox).  
4. The system will capture a configurable number of samples automatically (num\_samples, default 5). Ensure your face is clearly visible and still during capture.  
5. Verification will show success/failure in the "Application Results & Logs" area.

### **Authentication Flow**

1. Navigate to the "Main Controls" tab.  
2. Stand in front of the camera.  
3. Click "Login Person".  
4. Recognition results and match distance will appear in the "Application Results & Logs" area and the status label.  
5. **Adjust the "Recognition Threshold" slider** for desired sensitivity. A lower value (e.g., 0.1) requires higher similarity for a match, while a higher value (e.g., 0.4) is more lenient.

### **Settings Management**

1. Navigate to the "Settings" tab.  
2. **Camera Settings:**  
   * Select "Camera Type": "Built-in Cam" or "ESP32 Cam".  
   * If "ESP32 Cam" is selected, enter its stream URL (e.g., http://192.168.1.5/cam-hi.jpg).  
   * Click "Apply Camera Settings" to switch.  
3. **Face Detection Model:**  
   * Select "Model Type": "YuNet Detector", "Haar Cascade Detector", or "Custom CNN Detector".  
   * Click "Apply Model Settings" to switch the active face detector.

### **Manage Persons**

1. Navigate to the "Manage Persons" tab.  
2. View a list of all registered persons, their names, access status, and registration date.  
3. Click the "Delete" button next to a person's entry to remove them from the database.  
4. Click "Refresh Person List" if the list doesn't update automatically after a registration or deletion.

## **License**

This project is licensed under the MIT License.