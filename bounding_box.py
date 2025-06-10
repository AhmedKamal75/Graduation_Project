import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

class BoundingBoxDetector:
    def __init__(self, cascade_path=None):
        
        if cascade_path is None:
            # needs opencv-python-headless, but opencv-python-headless dosn't have gui support.
            # cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def predict_bounding_box(self, frame):
        """
        Detect faces in a given frame.
        Args:
            frame (numpy.ndarray): The input image frame.
        Returns:
            list: List of bounding boxes [(x, y, w, h), ...]
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast (optional, but can improve detection)
        gray_frame = cv2.equalizeHist(gray_frame)
        
        # Calculate dynamic min and max sizes for face detection (2% and 95% of smallest frame dimension)
        height, width = gray_frame.shape
        min_size = int(min(height, width) * 0.02)
        max_size = int(min(height, width) * 0.95)


        # Detect faces using the profile face cascade classifier
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.05,  # Adjust as needed
            minNeighbors=5,      # Adjust as needed
            minSize=(min_size, min_size),  # Dynamic minimum size
            maxSize=(max_size, max_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )   


        if len(faces) == 0:
            return []
        x, y, w, h = faces[0]
        bbox = (x, y, x+w, y+h)
        
        return [bbox]
        
    def crop_face(self, frame, bbox):
        """
        Crop a face from the frame based on the bounding box.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x, y, w, h).
        Returns:
            numpy.ndarray: Cropped face image.
        """
        x, y, w, h = bbox
        cropped_face = frame[y:y+h, x:x+w]
        return cropped_face
    
    def draw_bounding_box(self, frame, bbox):
        """
        Draw a bounding box on the frame.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x, y, w, h).
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
# V5
class FaceBBoxModel(nn.Module):
    def __init__(self):
        super(FaceBBoxModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for bounding box prediction
        self.fc_bbox = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),  # Input size matches the flattened conv layer output
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: [x1, y1, width, height]
        )

    def forward(self, x):
        # Convolutional layers with Batch Norm and Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 7 * 7)

        # Bounding box prediction
        bbox_output = self.fc_bbox(x)

        return bbox_output

def save_model(model, path):
  """Saves the model's state dictionary to the specified path.

  Args:
    model: The PyTorch model to save.
    path: The path where the model will be saved.
  """
  torch.save(model.state_dict(), path)
  print(f"Model saved to {path}")

def load_model(model, path, device):
    """Loads the model's state dictionary from the specified path.

    Args:
      model: The PyTorch model to load the state dictionary into.
      path: The path where the model is saved.
      device: The device to load the model onto (e.g., 'cuda' or 'cpu').
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)  # Explicitly load the state_dict
    model.load_state_dict(state_dict)
    # print(f"Model loaded from {path}")
    return model

class BoundingBoxPredictor:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.load_model()
        
    def load_model(self):
        print(f"Loading bounding box model from {self.model_path}")
        print(f"Running on device: {self.device}")
        self.model = FaceBBoxModel().to(self.device)
        self.model = load_model(self.model, self.model_path, self.device)
        self.model.eval()
        print("Bounding box model loaded.")
        
    def preprocess_frame(self, frame):
        """Preprocess a frame for the model.

        Args:
            frame (numpy.ndarray): The input image frame.

        Returns:
            torch.Tensor: The preprocessed frame.
        """
        frame = cv2.resize(frame, (224, 224))  # Resize to match model input size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
        frame = frame / 255.0  # Normalize to [0, 1]
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return frame
    
    def predict_bounding_box(self, frame):
        """Predict the bounding box for a face in a given frame.

        Args:
            frame (numpy.ndarray): The input image frame.

        Returns:
            list: List of bounding box coordinates [x1, y1, x2, y2].
        """
        hight, width, _ = frame.shape
        frame = self.preprocess_frame(frame)
        with torch.no_grad():
            output = self.model(frame.to(self.device))
            output = output.cpu().numpy().flatten()
            x, y, w, h = output
            x = (x / 224) * width
            y = (y / 224) * hight
            w = (w / 224) * width
            h = (h / 224) * hight
            bbox = [int(x), int(y), int(x+w), int(y+h)]
            return [bbox]
        
    def draw_bounding_box(self, frame, bbox):
        """
        Draw a bounding box on the frame.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x, y, w, h).
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
    
    def crop_face(self, frame, bbox):
        """
        Crop a face from the frame based on the bounding box.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x, y, w, h).
        Returns:
            numpy.ndarray: Cropped face image.
        """
        x1, y1, x2, y2 = bbox
        cropped_face = frame[y1:y2, x1:x2]
        return cropped_face

class Counter():
    def __init__(self, start=0, end=100, step=1):
        self.start = start
        self.end = end
        self.step = step
        self.count = start
    def increment(self):
        self.count = (self.count + self.step) % self.end
        return self.count
    def decrement(self):
        self.count = (self.count - self.step)
        if self.count < self.start:
            self.count = self.end - self.step
        return self.count

def test():
    bbox_predictor = BoundingBoxPredictor(model_path="models/bbox_models/v5/bbox_v5_randomly_augmented_epoch_3.pth", device="cpu")
    counter = Counter(start=0, end=49, step=1) # end=49 means images 0 to 48
    running = True
    image_base_path = "images/Alexandra Daddario/Alexandra Daddario_"
    image_ext = ".jpg"

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  # Create window once
    cv2.resizeWindow("frame", 1000, 900)        # Resize window once

    while running:
        image_path = f"{image_base_path}{counter.count}{image_ext}"
        print(f"Loading image: {image_path}")

        frame = cv2.imread(image_path)
        display_image = None

        if frame is None:
            print(f"Error: Could not read image at {image_path}. Skipping.")
            display_image = np.zeros((600, 800, 3), dtype=np.uint8) # Blank image for error
            cv2.putText(display_image, f"Error loading: {os.path.basename(image_path)}", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            original_frame_for_crop = frame.copy()
            frame_for_display = frame.copy()

            bboxes = bbox_predictor.predict_bounding_box(original_frame_for_crop)

            if bboxes:
                bbox = bboxes[0]  # Process the first detected face

                x1_orig, y1_orig, x2_orig, y2_orig = bbox
                h_img, w_img, _ = original_frame_for_crop.shape
                x1 = max(0, x1_orig)
                y1 = max(0, y1_orig)
                x2 = min(w_img, x2_orig)
                y2 = min(h_img, y2_orig)

                if x2 > x1 and y2 > y1: # Check for valid bbox after clamping
                    cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cropped_face = original_frame_for_crop[y1:y2, x1:x2]

                    if cropped_face is not None and cropped_face.size > 0:
                        try:
                            frame_height_display = frame_for_display.shape[0]
                            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                                target_height = frame_height_display
                                aspect_ratio = cropped_face.shape[1] / cropped_face.shape[0]
                                target_width = int(target_height * aspect_ratio)
                                
                                cropped_face_resized = cv2.resize(cropped_face, (target_width, target_height))
                                display_image = np.concatenate((frame_for_display, cropped_face_resized), axis=1)
                            else:
                                print(f"Warning: Cropped face for {image_path} has zero dimension. Displaying frame with bbox.")
                                display_image = frame_for_display
                        except cv2.error as e:
                            print(f"OpenCV error during resize/concat for {image_path}: {e}")
                            display_image = frame_for_display
                        except Exception as e:
                            print(f"Unexpected error during resize/concat for {image_path}: {e}")
                            display_image = frame_for_display
                    else:
                        print(f"Warning: Cropped face is empty for {image_path}. Displaying frame with bbox.")
                        display_image = frame_for_display
                else:
                    print(f"Warning: Invalid bounding box after clamping for {image_path}. Displaying original frame.")
                    display_image = frame_for_display
            else:
                print(f"No face detected in {image_path}.")
                display_image = frame_for_display

        if display_image is not None:
            cv2.imshow("frame", display_image)
        else: # Should not happen with current logic, but as a fallback
            error_fallback_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(error_fallback_img, "Error displaying image", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", error_fallback_img)

        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for a key press

        if key == ord('q') or key == 27 or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1: # q, Esc, or window closed
            cv2.destroyAllWindows()
            running = False
        elif key == ord('a'):  # 'a' for previous/decrement
            counter.decrement()
        elif key == ord('d'):  # 'd' for next/increment
            counter.increment()
        # Any other key will re-evaluate the loop with the current (or new) counter value

    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    """
    this model is used to predict bounding box for a face in an image.
    and is trained on 224x224 images, from celebA dataset, which is a dataset of celebrity faces, and these images are of 
    224x224 size, and high quality, clear frontal face images.
    but this model is primarily used to detect faces from images produced by esp32-cam, which are of low quality, and 
    not clear enough.
    
    so, this model is not used in the final implementation, but is used for testing purposes.
    and we will use YuNet model to detect faces in the final implementation.
    but this model is still a good model, and can be used for face detection in high quality images.
    and we will try to improve this model in the future, by training it on low quality images.
    """
    test()
