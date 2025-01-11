import cv2
import numpy as np
import time

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

    while running:

        # image_path = f"images/Alexandra Daddario/Alexandra Daddario_{counter.count}.jpg"
        image_path = f"images/Zac Efron/Zac Efron_{counter.count}.jpg"
        frame = cv2.imread(image_path)
        bboxes = yunet_detector.predict_bounding_box(frame)
        for bbox in bboxes:
            cropped_face = yunet_detector.crop_face(frame, bbox)
            frame = yunet_detector.draw_bounding_box(frame, bbox)
            # Show frame with bounding box and cropped face concatenated horizontally
            frame_height = frame.shape[0]
            cropped_face_resized = cv2.resize(
                cropped_face,
                (int(cropped_face.shape[1] * (frame_height / cropped_face.shape[0])), frame_height)
            )
            display_frame = np.concatenate((frame, cropped_face_resized), axis=1)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 1000, 900)
            cv2.imshow("frame", display_frame)
            
        # Wait 'q', 'Esc', or close button is pressed
        key = cv2.waitKey(1) & 0xFF
        while key == 255:
            time.sleep(0.001)
            key = cv2.waitKey(1) & 0xFF
            
        if key == ord('q') or key == 27 or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            running = False
            
        # if left arrow key is pressed then decrement the counter
        elif key == 81:
            counter.decrement()
            running = True
            
        # if right arrow key is pressed then increment the counter
        elif key == 83:
            counter.increment()
            running = True
            
        else:
            running = True
            
    cv2.destroyAllWindows()
    return
    

if __name__ == "__main__":
    test()
