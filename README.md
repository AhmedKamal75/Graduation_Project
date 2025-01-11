# Face Recognition System

A robust facial recognition application built with Python, utilizing custom trained deep learning models for face detection and recognition.



## Features

- Real-time face detection and recognition
- Custom-trained models
- User registration and authentication system
- Access control management
- GUI interface built with CustomTkinter
- Support for IP camera integration
- Queue-based embedding averaging for better recognition
- Thread-safe database operations

## Technical Stack

- **Language:** Python
- **GUI Framework:** CustomTkinter
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV
- **Database:** SQLite
- **Camera Interface:** ESP32-CAM

## Models

### Face Detection
- Custom trained model on CelebA dataset
- Model architecture: CNN with multiple convolutional and fully connected layers
- Input size: 224x224 pixels
- Output: 4 coordinates (x, y, width, height)
- Training dataset: CelebA with 200k+ images https://huggingface.co/datasets/hfaus/CelebA_bbox_and_facepoints
- Training notebook: models/bbox_models/v5/facial-detection-celebA-bbox-optimized.ipynb
- Kaggle Training notebook: https://www.kaggle.com/code/ahmedkamal75/facial-detection-celeba-bbox-optimized
- Models : 
    - Without Augmentations:
        - https://www.kaggle.com/models/ahmedkamal75/bbox_model_v5_epoch_8/
    - With Augmentations:
        - https://www.kaggle.com/models/ahmedkamal75/bbox_v5_augmented_epoch_50




### Face Recognition
- ArcFace-based architecture with residual connections
- Embedding dimension: 512
- Loss function: ArcFace loss with s=30, m=0.5 for the first 40 epochs, and then s=45, m=0.7 for the remaining epochs
- Preprocessing: Face alignment using detected landmarks
- Training dataset: https://www.kaggle.com/datasets/zenbot99/vggface2-hq-cropped (Kaggle)
- Testing dataset: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition (Kaggle)
- Training notebook: `models/resarksgd/arkface-residual-connections-part-2.ipynb`
- Kaggle Training notebook: https://www.kaggle.com/code/ahmedkamal75/arkface-residual-connections-part-2
- Models: 
    - not augmented 88% accuracy: https://www.kaggle.com/models/ahmedkamal75/resarksgd-acc-88/
    - not augmented 95% accuracy: https://www.kaggle.com/models/ahmedkamal75/resarksgd95/
    - augmented 94% accuracy: https://www.kaggle.com/code/ahmedkamal75/arkface-residual-connections-part-2/notebook?scriptVersionId=217035765

## Requirements
- Python 3.8+
- CUDA capable GPU (recommended)
- ESP32-CAM
- Minimum 4GB RAM
- Storage: 500MB for models


## Project Structure
```
Graduation_Project/
├── app.py # Main application using custom models. 
├── insightface_app.py # Alternative implementation using InsightFace 
├── embeddings.py # Face recognition model implementation 
├── bounding_box.py # Face detection model implementation 
├── bounding_box_yunet.py # YuNet face detector implementation 
└── models/ 
    ├── bbox_models/ # Face detection models 
        ├── v5/ # Custom trained models
        ├── YuNet/ # YuNet face detector models
    └── resarksgd/ # Face recognition models
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AhmedKamal75/Graduation_Project.git
    cd Graduation_Project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .venv      # Using .venv is a common convention
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the resarksgd95.pth model (or the desired version) from: Kaggle Model
    
    [model with accuracy of 95%](https://www.kaggle.com/models/ahmedkamal75/resarksgd95/)

    - Place the downloaded resarksgd95.pth file in the models/resarksgd/ directory.

    - Important: Update the embedding_predictor in the application code (app.py) to point to the correct path of the model file. For example:
    ```
    embedding_predictor = EmbeddingPredictor(model_path='models/resarksgd/resarksgd95.pth', device='cpu')
    # Or the name of your downloaded model
    ```
    


## Usage Examples
### Registration Process
1. Run the main application:
    ```bash
    python app.py
    ```
2. Click "Register Person"
3. Enter name and set access permissions
4. System will capture 10 samples automatically
5. Verification will show success/failure

### Authentication Flow
1. Stand in front of camera
2. Click "Login Person"
3. Recognition results appear

### Note:
- Adjust the threshold to the proper value (0.1 if possible)


## License

This project is licensed under the MIT License.