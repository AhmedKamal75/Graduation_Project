from bounding_box import BoundingBoxDetector, BoundingBoxPredictor
from bounding_box_yunet import YuNetFaceDetector
from embeddings import EmbeddingPredictor
import cv2
import urllib.request
import numpy as np

# Example usage
def process_frame(camera_url, bbox_predictor=None, embedding_predictor=None):
    # Capture frame from camera
    img_resp = urllib.request.urlopen(camera_url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    # Detect faces
    faces = bbox_predictor.predict_bounding_box(frame)
    for bbox in faces:
        # Crop face
        cropped_face = bbox_predictor.crop_face(frame, bbox)

        # Draw bounding box
        frame = bbox_predictor.draw_bounding_box(frame, bbox)
        
        # Generate embedding
        if embedding_predictor is not None:
            embedding = embedding_predictor.generate_embedding(cropped_face)
            if embedding is not None:
                embedding = embedding.flatten()
                print(f"Generated embedding Shape: {embedding.shape}")


    frame_height = frame.shape[0]
    cropped_face_resized = cv2.resize(cropped_face, (int(cropped_face.shape[1] * (frame_height / cropped_face.shape[0])), frame_height))
    frame = np.concatenate((frame, cropped_face_resized), axis=1)
    cv2.namedWindow("frame and cropped face", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame and cropped face", 1000, 900)
    cv2.imshow("frame and cropped face", frame)

    # Wait 'q', 'Esc', or close button is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 or cv2.getWindowProperty("frame and cropped face", cv2.WND_PROP_VISIBLE) < 1: # checks if 'q', 'Esc', or close button is pressed
        cv2.destroyAllWindows()
        return False
    else:
        return True
    


if __name__ == "__main__":
    camera_url = "http://192.168.1.5/cam-hi.jpg"
    
    # Initialize modules
    # bbox_predictor = BoundingBoxDetector()
    bbox_predictor = BoundingBoxPredictor(model_path="models/bbox_models/v5/bbox_v5_randomly_augmented_epoch_3.pth", device="cpu") # bad in absence of light and not accurate and only detects if the face is fully visible and centered in the frame 
    # embedding_predictor = EmbeddingPredictor(model_path="models/resarksgd/resarksgd95.pth", device="cpu")
    embedding_predictor = None


    while True:
        # process_frame(camera_url)
        if not process_frame(
            camera_url=camera_url,
            bbox_predictor=bbox_predictor,
            embedding_predictor=embedding_predictor
        ):
            break
    
        
        
