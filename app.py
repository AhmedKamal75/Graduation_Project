import cv2
import numpy as np
import threading
import os
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import sqlite3
from queue import Queue
import time
import logging
import urllib.request
from embeddings import EmbeddingPredictor
from bounding_box import BoundingBoxDetector, BoundingBoxPredictor
from bounding_box_yunet import YuNetFaceDetector


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)

class PersonDatabase:
    def __init__(self, db_file='face_database.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    embedding BLOB,
                    is_allowed INTEGER,
                    created_at TEXT
                )
            ''')

    def add_person(self, embedding, name, is_allowed):
        with self.lock:
            person_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            with self.conn:
                self.conn.execute(
                    'INSERT INTO persons (id, name, embedding, is_allowed, created_at) VALUES (?, ?, ?, ?, ?)',
                    (person_id, name, sqlite3.Binary(embedding.tobytes()), int(is_allowed), datetime.now().isoformat())
                )
            return person_id

    def find_match(self, embedding, threshold=0.3):
        with self.lock:
            cursor = self.conn.execute('SELECT id, name, embedding, is_allowed FROM persons')
            min_distance = float('inf')
            best_match = None

            for row in cursor:
                db_embedding = np.frombuffer(row[2], dtype=np.float32)
                # using cosine similarity
                similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                distance = 1 - similarity # convert to distance, as cosine similarity is 1 for identical vectors and 0 for opposite vectors
                # using euclidean distance
                # distance = np.linalg.norm(embedding - db_embedding)
                logging.info(f"Distance to {row[1]}: {distance}")

                if distance < min_distance:
                    min_distance = distance
                    best_match = row

            if min_distance < threshold:
                return best_match[0], {'name': best_match[1], 'is_allowed': bool(best_match[3])}, min_distance
            return None, None, min_distance

class FaceRecognitionApp:
    def __init__(self, camera_url, database_path, embedding_predictor, bbox_predictor, threshold=0.3, num_samples=5):
        self.camera_url = camera_url
        self.person_db = PersonDatabase(database_path)
        self.latest_face = None
        self.latest_bbox = None
        self.processing = False
        self.threshold = threshold
        self.num_samples = num_samples
        self.embedding_queue = Queue(maxsize=self.num_samples) 
        self.lock = threading.Lock()

        self.embedding_predictor = embedding_predictor
        self.bbox_predictor = bbox_predictor

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Recognition System Using Insightface")
        self.root.geometry("1200x800")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # Video Feed Panel
        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Video Feed", anchor="center")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Control Panel
        self.control_panel = ctk.CTkFrame(self.root)
        self.control_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Registration Section
        self.reg_frame = ctk.CTkFrame(self.control_panel)
        self.reg_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.reg_frame, text="Registration", font=("Arial", 16)).pack(padx=5, pady=5)

        ctk.CTkLabel(self.reg_frame, text="Name:").pack(padx=5, pady=2)
        self.name_entry = ctk.CTkEntry(self.reg_frame)
        self.name_entry.pack(fill="x", padx=5, pady=2)

        self.is_allowed_var = ctk.BooleanVar()
        ctk.CTkCheckBox(self.reg_frame, text="Allowed Access", variable=self.is_allowed_var).pack(padx=5, pady=2)

        self.register_btn = ctk.CTkButton(self.reg_frame, text="Register Person", command=self.register_person)
        self.register_btn.pack(fill="x", padx=5, pady=5)

        # Login Section
        self.login_frame = ctk.CTkFrame(self.control_panel)
        self.login_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.login_frame, text="Login", font=("Arial", 16)).pack(padx=5, pady=5)

        self.login_btn = ctk.CTkButton(self.login_frame, text="Login Person", command=self.login_person)
        self.login_btn.pack(fill="x", padx=5, pady=5)

        # Results Section
        self.results_frame = ctk.CTkFrame(self.control_panel)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(self.results_frame, text="Results", font=("Arial", 16)).pack(padx=5, pady=5)

        self.results_text = ctk.CTkTextbox(self.results_frame, height=10)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Threshold Section
        self.threshold_frame = ctk.CTkFrame(self.control_panel)
        self.threshold_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.threshold_frame, text="Threshold Settings", font=("Arial", 16)).pack(padx=5, pady=5)

        self.threshold_slider = ctk.CTkSlider(self.threshold_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_threshold)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(fill="x", padx=5, pady=5)

        self.threshold_label = ctk.CTkLabel(self.threshold_frame, text=f"Threshold: {self.threshold:.3f}")
        self.threshold_label.pack(padx=5, pady=5)

        self.min_distance_label = ctk.CTkLabel(self.threshold_frame, text="Last Min Distance: N/A")
        self.min_distance_label.pack(padx=5, pady=5)

        # Exit Button
        self.exit_btn = ctk.CTkButton(self.control_panel, text="Exit Program", command=self.exit_program)
        self.exit_btn.pack(fill="x", padx=10, pady=10)

        self.running = True
        self.capture_thread = None
        self.start_capture()

    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_label.configure(text=f"Threshold: {self.threshold:.5f}")

    def extract_face(self, frame):
        faces = self.bbox_predictor.predict_bounding_box(frame)
        if len(faces) > 0:
            self.latest_bbox = faces[0]
            cropped_face = self.bbox_predictor.crop_face(frame, faces[0])
            # return self.embedding_predictor.generate_embedding(cropped_face).flatten()
            embedding = self.embedding_predictor.generate_embedding(cropped_face).flatten()

            with self.lock:
                if self.embedding_queue.full():
                    self.embedding_queue.get()
                self.embedding_queue.put(embedding)

            return embedding
        self.latest_bbox = None
        return None

    def capture_multiple_embeddings(self, num_samples=10):
        """Get the average of recent embeddings from the queue."""
        if num_samples is None:
            num_samples = self.num_samples
            
        # Wait until we have enough samples
        timeout = 5  # seconds to wait for samples
        start_time = time.time()
        while self.embedding_queue.qsize() < num_samples:
            if time.time() - start_time > timeout:
                if self.embedding_queue.qsize() == 0:
                    return None
                break
            time.sleep(0.1)
        
        # Get all available embeddings from queue
        embeddings = []
        with self.lock:
            while not self.embedding_queue.empty():
                embeddings.append(self.embedding_queue.get())
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
        
    def start_capture(self):
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.update_frame()

    def capture_frames(self):
        while self.running:
            try:
                img_resp = urllib.request.urlopen(self.camera_url)
                img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)
                self.latest_face = self.extract_face(frame)

                if self.latest_bbox is not None:
                    x1, y1, x2, y2 = self.latest_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = Image.fromarray(rgb_frame)

            except Exception as e:
                logging.error(f"Frame capture error: {e}")

    def update_frame(self):
        if hasattr(self, 'current_frame'):
            display_size = (960, 720)
            resized_frame = self.current_frame.resize(display_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=resized_frame)
            self.video_label.configure(image=self.photo)

        if self.running: # 
            self.root.after(30, self.update_frame) 

    def register_person(self):
        if self.processing:
            messagebox.showwarning("Processing", "Please wait, still processing previous request")
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        def register_thread():
            try:
                self.processing = True
                self.register_btn.configure(state='disabled')
                self.login_btn.configure(state='disabled')

                embedding = self.capture_multiple_embeddings(num_samples=self.num_samples)
                if embedding is None:
                    messagebox.showerror("Error", "Failed to extract face features")
                    return

                person_id = self.person_db.add_person(
                    embedding, name, self.is_allowed_var.get()
                )
                logging.info(f"Registered {name} with ID {person_id}")
                self.results_text.insert('1.0', f"Registered: {name}\n")
                self.name_entry.delete(0, ctk.END)
            except Exception as e:
                messagebox.showerror("Error", f"Registration failed: {e}")
            finally:
                self.processing = False
                self.register_btn.configure(state='normal')
                self.login_btn.configure(state='normal')

        threading.Thread(target=register_thread).start()
        
    def login_person(self):        
        if self.processing:
            messagebox.showwarning("Processing", "Please wait, still processing previous request")
            return

        def login_thread():
            try:
                self.processing = True
                self.register_btn.configure(state='disabled')
                self.login_btn.configure(state='disabled')

                embedding = self.capture_multiple_embeddings(num_samples=self.num_samples)
                if embedding is None:
                    messagebox.showerror("Error", "Failed to extract face features")
                    return

                person_id, person_data, min_distance = self.person_db.find_match(embedding, self.threshold)

                self.min_distance_label.configure(text=f"Last Min Distance: {min_distance:.5f}")

                if person_data:
                    result = (
                        f"Login Result:\n"
                        f"Name: {person_data['name']}\n"
                        f"Access: {'Allowed' if person_data['is_allowed'] else 'Denied'}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'-'*30}\n"
                    )
                else:
                    result = "No match found\n"

                self.results_text.insert('1.0', result)
            except Exception as e:
                messagebox.showerror("Error", f"Login failed: {e}")
            finally:
                self.processing = False
                self.register_btn.configure(state='normal')
                self.login_btn.configure(state='normal')

        threading.Thread(target=login_thread).start()

    def exit_program(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.running = False
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            self.root.quit()
            self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
        self.root.mainloop()


if __name__ == "__main__":
    esp32_cam_url = "http://192.168.1.5/cam-hi.jpg"
    database_path = "face_database.db"
    embedding_predictor = EmbeddingPredictor(model_path='models/resarksgd/resarksgdaug94.pth', device='cpu')
    # bbox_predictor = BoundingBoxDetector()
    # bbox_predictor = BoundingBoxPredictor(model_path="models/bbox_models/v5/bbox_v5_randomly_augmented_epoch_3.pth", device="cpu")
    bbox_predictor = YuNetFaceDetector(model_path="models/bbox_models/YuNet/face_detection_yunet_2023mar.onnx", input_size=(320, 320), conf_threshold=0.9, nms_threshold=0.3, top_k=5000)

    app = FaceRecognitionApp(
        camera_url=esp32_cam_url, # camera URL
        database_path=database_path, # path to the database file
        embedding_predictor=embedding_predictor, # embedding predictor object
        bbox_predictor=bbox_predictor, # bounding box predictor object
        threshold=0.25, # threshold for cosine similarity
        num_samples=1 # number of samples to average for each embedding
    )
    app.run()


