import cv2
import numpy as np
import time
import os # Added for os.path.basename

class YuNetFaceDetector():
    """
    YuNetFaceDetector class for face detection using YuNet model.

    Args:
        model_path (str): Path to the YuNet model file.
        input_size (tuple): Input size for the model (width, height).
        conf_threshold (float): Confidence threshold for face detection.
        nms_threshold (float): Non-maximum suppression threshold for face detection.
        top_k (int): Number of top faces to keep after non-maximum suppression.
    """
    def __init__(self, model_path, input_size=(320, 320), conf_threshold=0.9, nms_threshold=0.3, top_k=5000):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.load_model()

    def load_model(self):
        self.face_detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self.input_size,
            self.conf_threshold,
            self.nms_threshold,
            self.top_k
        )

    def predict_bounding_box(self, frame):
        """
        Detect faces in a given frame.
        Args:
            frame (numpy.ndarray): The input image frame.
        Returns:
            list: List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        height, width, _ = frame.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        if faces is None:
            return []
        bboxes = []
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            bboxes.append((x, y, x + w, y + h))
        return bboxes

    def crop_face(self, frame, bbox):
        """
        Crop a face from the frame based on the bounding box.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x1, y1, x2, y2).
        Returns:
            numpy.ndarray: Cropped face image.
        """
        x1, y1, x2, y2 = bbox
        cropped_face = frame[y1:y2, x1:x2]
        return cropped_face

    def draw_bounding_box(self, frame, bbox):
        """
        Draw a bounding box on the frame.
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (tuple): Bounding box (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

class Counter():
    """
    A simple counter class.
    
    Args:
        start (int): The starting value of the counter.
        end (int): The ending value of the counter.
        step (int): The step value of the counter.
    Returns:
        int: The current value of the counter.
    """
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
    """
        Test the YuNetFaceDetector class.

        Usage:
            python bounding_box_yunet.py
        
        Press 'q' or 'Esc' to exit the program.
        Press left arrow key to decrement the counter.
        Press right arrow key to increment the counter.
        Press any other key to continue.

        Returns:
            None
    """

    yunet_detector = YuNetFaceDetector(
        model_path="models/bbox_models/YuNet/face_detection_yunet_2023mar.onnx"
    )
    counter = Counter(start=30, end=35, step=1)
    running = True

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  # Create window once
    cv2.resizeWindow("frame", 1000, 900)        # Resize window once

    while running:
        # image_path = f"images/Alexandra Daddario/Alexandra Daddario_{counter.count}.jpg"
        image_path = f"images/Zac Efron/Zac Efron_{counter.count}.jpg"
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

            bboxes = yunet_detector.predict_bounding_box(original_frame_for_crop)

            if bboxes:
                bbox = bboxes[0]  # Process the first detected face

                x1_orig, y1_orig, x2_orig, y2_orig = bbox
                h_img, w_img, _ = original_frame_for_crop.shape
                x1 = max(0, x1_orig)
                y1 = max(0, y1_orig)
                x2 = min(w_img, x2_orig)
                y2 = min(h_img, y2_orig)

                if x2 > x1 and y2 > y1: # Check for valid bbox after clamping
                    yunet_detector.draw_bounding_box(frame_for_display, (x1, y1, x2, y2))
                    cropped_face = yunet_detector.crop_face(original_frame_for_crop, (x1, y1, x2, y2))

                    if cropped_face is not None and cropped_face.size > 0:
                        try:
                            frame_height_display = frame_for_display.shape[0]
                            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0: # Ensure valid dimensions
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
        else: # Fallback if display_image somehow remains None
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
    test()
