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
import urllib.request # For IP Camera functionality

from embeddings import EmbeddingPredictor
# Import all BoundingBoxDetector types for selection
from bounding_box import BoundingBoxDetector, BoundingBoxPredictor # BoundingBoxPredictor is the custom CNN
from bounding_box_yunet import YuNetFaceDetector


logging.basicConfig(
    # Corrected logging format string
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)

class PersonDatabase:
    """Manages person data and embeddings storage in an SQLite database."""
    def __init__(self, db_file='face_database.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.lock = threading.Lock() # For thread-safe database access
        self._create_table()

    def _create_table(self):
        """Creates the 'persons' table if it doesn't exist."""
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
        """Adds a new person's data and embedding to the database."""
        with self.lock:
            person_id = datetime.now().strftime("%Y%m%d_%H%M%S") # Unique ID based on timestamp
            with self.conn:
                self.conn.execute(
                    'INSERT INTO persons (id, name, embedding, is_allowed, created_at) VALUES (?, ?, ?, ?, ?)',
                    (person_id, name, sqlite3.Binary(embedding.tobytes()), int(is_allowed), datetime.now().isoformat())
                )
            logging.info(f"Added person '{name}' with ID '{person_id}' to database.")
            return person_id

    def find_match(self, embedding, threshold=0.3):
        """
        Finds the closest matching person in the database for a given embedding.
        Returns the person's ID, data, and the minimum distance if a match is found within the threshold.
        """
        with self.lock:
            cursor = self.conn.execute('SELECT id, name, embedding, is_allowed FROM persons')
            min_distance = float('inf')
            best_match = None

            for row in cursor:
                db_embedding = np.frombuffer(row[2], dtype=np.float32)
                # Calculate cosine distance: 0 for identical, 2 for opposite.
                similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                distance = 1 - similarity 
                logging.debug(f"Distance to {row[1]}: {distance:.5f}") # Use debug for frequent logs

                if distance < min_distance:
                    min_distance = distance
                    best_match = row

            if min_distance < threshold:
                # If the closest distance is below the threshold, consider it a match.
                logging.info(f"Match found: {best_match[1]} with distance {min_distance:.5f} (Threshold: {threshold:.3f})")
                return best_match[0], {'name': best_match[1], 'is_allowed': bool(best_match[3])}, min_distance
            
            # No match found within the threshold
            logging.info(f"No match found. Closest was {min_distance:.5f} (Threshold: {threshold:.3f})")
            return None, None, min_distance
    
    def get_all_persons(self):
        """Retrieves all registered persons from the database."""
        with self.lock:
            cursor = self.conn.execute('SELECT id, name, is_allowed, created_at FROM persons ORDER BY created_at DESC')
            persons = []
            for row in cursor:
                persons.append({
                    'id': row[0],
                    'name': row[1],
                    'is_allowed': bool(row[2]),
                    'created_at': row[3]
                })
            return persons
            
    def delete_person(self, person_id):
        """Deletes a person from the database by ID."""
        with self.lock:
            cursor = self.conn.execute('DELETE FROM persons WHERE id = ?', (person_id,))
            self.conn.commit()
            return cursor.rowcount > 0 # Returns True if a row was deleted, False otherwise


class FaceRecognitionApp:
    """Main application class for the Face Recognition System GUI."""
    def __init__(self, database_path, embedding_predictor,
                 initial_camera_index=0, initial_ip_cam_url="http://192.168.1.5/cam-hi.jpg",
                 initial_camera_type="Built-in Cam", 
                 initial_bbox_model_type="YuNet Detector", # New: default bounding box model
                 yunet_model_path="models/bbox_models/YuNet/face_detection_yunet_2023mar.onnx", # YuNet model path
                 haar_cascade_path=None, # Haar cascade default path (OpenCV's default)
                 custom_cnn_model_path="models/bbox_models/v5/bbox_v5_randomly_augmented_epoch_3.pth", # New: Custom CNN model path
                 threshold=0.25, num_samples=5):
        
        self.person_db = PersonDatabase(database_path)
        self.embedding_predictor = embedding_predictor
        
        # Camera management variables
        self.cap = None  # OpenCV VideoCapture object for built-in camera
        self.built_in_camera_index = initial_camera_index
        self.ip_cam_url = initial_ip_cam_url
        self.active_camera_type = initial_camera_type # "Built-in Cam" or "IP Cam"
        
        # Bounding box predictor management variables
        self.yunet_model_path = yunet_model_path
        # Use default Haar cascade path if not provided
        self.haar_cascade_path = haar_cascade_path if haar_cascade_path else cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.custom_cnn_model_path = custom_cnn_model_path # Store path for Custom CNN
        self.active_bbox_model_type = initial_bbox_model_type
        self.bbox_predictor = None # Initialize as None, will be set by _initialize_bbox_predictor

        # Face detection and recognition related
        self.latest_face = None
        self.latest_bbox = None
        self.processing = False # Flag to prevent concurrent registration/login operations
        self.threshold = threshold # Cosine distance threshold for recognition
        self.num_samples = num_samples # Number of face samples to average for robust embedding
        self.embedding_queue = Queue(maxsize=self.num_samples) # Queue for storing recent embeddings
        self.lock = threading.Lock() # Lock for thread-safe access to shared resources (e.g., queue)

        # CustomTkinter GUI setup
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Face Recognition System")
        self.root.geometry("1400x900") # Increased geometry for tabs
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3) # Video feed panel
        self.root.grid_columnconfigure(1, weight=1) # Control panel

        # --- Video Feed Panel ---
        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Video Feed", anchor="center")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Control Panel (now uses tabs) ---
        self.control_panel_frame = ctk.CTkFrame(self.root)
        self.control_panel_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.tab_view = ctk.CTkTabview(self.control_panel_frame)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.main_tab = self.tab_view.add("Main Controls")
        self.settings_tab = self.tab_view.add("Settings")
        self.manage_persons_tab = self.tab_view.add("Manage Persons")

        # Configure tabs grid/pack (each tab has its own layout)
        self.main_tab.grid_columnconfigure(0, weight=1)
        self.settings_tab.grid_columnconfigure(0, weight=1)
        self.manage_persons_tab.grid_columnconfigure(0, weight=1)


        # --- Main Controls Tab ---
        # Registration Section
        self.reg_frame = ctk.CTkFrame(self.main_tab)
        self.reg_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.reg_frame, text="Registration", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        ctk.CTkLabel(self.reg_frame, text="Name:").pack(padx=5, pady=2)
        self.name_entry = ctk.CTkEntry(self.reg_frame)
        self.name_entry.pack(fill="x", padx=5, pady=2)
        self.is_allowed_var = ctk.BooleanVar(value=True) # Default to allowed
        ctk.CTkCheckBox(self.reg_frame, text="Allowed Access", variable=self.is_allowed_var).pack(padx=5, pady=2)
        self.register_btn = ctk.CTkButton(self.reg_frame, text="Register Person", command=self.register_person)
        self.register_btn.pack(fill="x", padx=5, pady=5)

        # Login Section
        self.login_frame = ctk.CTkFrame(self.main_tab)
        self.login_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.login_frame, text="Login", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        self.login_btn = ctk.CTkButton(self.login_frame, text="Login Person", command=self.login_person)
        self.login_btn.pack(fill="x", padx=5, pady=5)
        
        # Threshold Section - Moved to Main Controls for easy access during login
        self.threshold_frame = ctk.CTkFrame(self.main_tab)
        self.threshold_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.threshold_frame, text="Recognition Threshold", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        self.threshold_slider = ctk.CTkSlider(self.threshold_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_threshold)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(fill="x", padx=5, pady=5)
        self.threshold_label = ctk.CTkLabel(self.threshold_frame, text=f"Current Threshold: {self.threshold:.3f}")
        self.threshold_label.pack(padx=5, pady=5)
        self.min_distance_label = ctk.CTkLabel(self.threshold_frame, text="Last Match Distance: N/A")
        self.min_distance_label.pack(padx=5, pady=5)

        # Results & Logs Section - Moved to Main Controls for immediate feedback
        self.results_logs_section_frame = ctk.CTkFrame(self.main_tab)
        self.results_logs_section_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(self.results_logs_section_frame, text="Application Results & Logs", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(self.results_logs_section_frame, text="", font=("Arial", 14), text_color="grey")
        self.status_label.pack(padx=5, pady=5)

        self.results_text = ctk.CTkTextbox(self.results_logs_section_frame, height=15) # Adjusted height
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._setup_gui_logger() # Override the default logging stream handler to direct messages here


        # --- Settings Tab ---
        # Camera Source Selection Group
        self.camera_settings_frame = ctk.CTkFrame(self.settings_tab)
        self.camera_settings_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.camera_settings_frame, text="Camera Settings", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        ctk.CTkLabel(self.camera_settings_frame, text="Camera Source:").pack(padx=5, pady=2)
        self.camera_options = ["Built-in Cam", "IP Cam"]
        self.camera_type_var = ctk.StringVar(value=self.active_camera_type)
        self.camera_type_optionmenu = ctk.CTkOptionMenu(
            self.camera_settings_frame,
            values=self.camera_options,
            command=self.on_camera_type_selected,
            variable=self.camera_type_var
        )
        self.camera_type_optionmenu.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.camera_settings_frame, text="IP Cam URL:").pack(padx=5, pady=2)
        self.ip_cam_url_entry = ctk.CTkEntry(self.camera_settings_frame)
        self.ip_cam_url_entry.insert(0, self.ip_cam_url) # Set initial URL
        self.ip_cam_url_entry.pack(fill="x", padx=5, pady=2)
        self.apply_camera_settings_btn = ctk.CTkButton(self.camera_settings_frame, text="Apply Camera Settings", command=self.apply_camera_settings)
        self.apply_camera_settings_btn.pack(fill="x", padx=5, pady=5)
        self._update_ip_cam_url_entry_state() # Initial state of IP Cam URL entry

        # Face Detection Model Selection Group
        self.bbox_model_settings_frame = ctk.CTkFrame(self.settings_tab)
        self.bbox_model_settings_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.bbox_model_settings_frame, text="Face Detection Model", font=("Arial", 18, "bold")).pack(padx=5, pady=5)
        ctk.CTkLabel(self.bbox_model_settings_frame, text="Model Type:").pack(padx=5, pady=2)
        # Added "Custom CNN Detector"
        self.bbox_model_options = ["YuNet Detector", "Haar Cascade Detector", "Custom CNN Detector"]
        self.bbox_model_type_var = ctk.StringVar(value=self.active_bbox_model_type)
        self.bbox_model_type_optionmenu = ctk.CTkOptionMenu(
            self.bbox_model_settings_frame,
            values=self.bbox_model_options,
            command=self.on_bbox_model_type_selected,
            variable=self.bbox_model_type_var
        )
        self.bbox_model_type_optionmenu.pack(fill="x", padx=5, pady=5)
        self.apply_bbox_model_btn = ctk.CTkButton(self.bbox_model_settings_frame, text="Apply Model Settings", command=self.apply_bbox_model_settings)
        self.apply_bbox_model_btn.pack(fill="x", padx=5, pady=5)


        # --- Manage Persons Tab ---
        self.manage_persons_content_frame = ctk.CTkFrame(self.manage_persons_tab)
        self.manage_persons_content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(self.manage_persons_content_frame, text="Manage Registered Persons", font=("Arial", 18, "bold")).pack(padx=5, pady=5)

        # Frame for the list header (fixed)
        self.person_list_header_frame = ctk.CTkFrame(self.manage_persons_content_frame)
        self.person_list_header_frame.pack(fill="x", padx=5, pady=(5,0))
        # Adjusted minsize for ID and increased weight for Action to ensure visibility
        self.person_list_header_frame.grid_columnconfigure(0, weight=1, minsize=100) # ID
        self.person_list_header_frame.grid_columnconfigure(1, weight=2, minsize=150) # Name
        self.person_list_header_frame.grid_columnconfigure(2, weight=1, minsize=80)  # Allowed
        self.person_list_header_frame.grid_columnconfigure(3, weight=2, minsize=180) # Registered At
        self.person_list_header_frame.grid_columnconfigure(4, weight=2, minsize=120) # Action (increased minsize/weight)
        
        ctk.CTkLabel(self.person_list_header_frame, text="ID", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        ctk.CTkLabel(self.person_list_header_frame, text="Name", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        ctk.CTkLabel(self.person_list_header_frame, text="Allowed", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=2, pady=2, sticky="w")
        ctk.CTkLabel(self.person_list_header_frame, text="Registered At", font=("Arial", 12, "bold")).grid(row=0, column=3, padx=2, pady=2, sticky="w")
        ctk.CTkLabel(self.person_list_header_frame, text="Action", font=("Arial", 12, "bold")).grid(row=0, column=4, padx=2, pady=2, sticky="w")

        # Use a CTkCanvas with scrollbars for the main list area
        self.person_list_canvas_container = ctk.CTkFrame(self.manage_persons_content_frame)
        self.person_list_canvas_container.pack(fill="both", expand=True, padx=5, pady=(0,5))
        self.person_list_canvas_container.grid_rowconfigure(0, weight=1)
        self.person_list_canvas_container.grid_columnconfigure(0, weight=1)

        # Get the background color from the theme for the canvas
        # This accesses CustomTkinter's internal theme dictionary
        try:
            current_mode = ctk.get_appearance_mode()
            mode_index = 0 if current_mode == "Light" else 1
            # More robust way to get the actual background color from the theme
            bg_color_canvas = ctk.ThemeManager.theme["CTkFrame"]["fg_color"][
                mode_index
            ]
        except AttributeError:
            # Fallback if the internal attribute is not directly accessible (less common)
            # Default to a known dark color for safety
            bg_color_canvas = "#2B2B2B" # A common dark mode background color for CTk

        self.person_list_canvas = ctk.CTkCanvas(
            self.person_list_canvas_container,
            highlightthickness=0,
            bg=bg_color_canvas, # Set standard Tkinter canvas background
            highlightbackground=bg_color_canvas, # Remove white highlight border
            borderwidth=0
        )
        self.person_list_canvas.grid(row=0, column=0, sticky="nsew")

        self.person_list_v_scrollbar = ctk.CTkScrollbar(self.person_list_canvas_container, orientation="vertical", command=self.person_list_canvas.yview)
        self.person_list_v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.person_list_canvas.configure(yscrollcommand=self.person_list_v_scrollbar.set)

        self.person_list_h_scrollbar = ctk.CTkScrollbar(self.person_list_canvas_container, orientation="horizontal", command=self.person_list_canvas.xview)
        self.person_list_h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.person_list_canvas.configure(xscrollcommand=self.person_list_h_scrollbar.set)

        self.person_list_inner_frame = ctk.CTkFrame(self.person_list_canvas, fg_color="transparent")
        # create_window places the frame inside the canvas
        self.person_list_canvas.create_window((0, 0), window=self.person_list_inner_frame, anchor="nw", tags="inner_frame")

        # Bind the inner frame's configure event to update the canvas scrollregion
        # This is crucial for the canvas to know the size of its content
        self.person_list_inner_frame.bind("<Configure>", self._on_inner_frame_configure)

        # Bind mouse wheel for scrolling (for the canvas) - Windows/macOS
        self.person_list_canvas.bind_all("<MouseWheel>", self._on_mousewheel_canvas)
        # Bind mouse wheel for scrolling (for Linux)
        self.person_list_canvas.bind_all("<Button-4>", self._on_mousewheel_canvas_linux)
        self.person_list_canvas.bind_all("<Button-5>", self._on_mousewheel_canvas_linux)

        self.person_row_frames = [] # To hold references to row frames for clearing

        self.refresh_persons_btn = ctk.CTkButton(self.manage_persons_content_frame, text="Refresh Person List", command=self._load_persons_to_gui)
        self.refresh_persons_btn.pack(fill="x", padx=5, pady=5)
        
        self._load_persons_to_gui() # Load persons on startup

        # --- Exit Button ---
        self.exit_btn = ctk.CTkButton(self.control_panel_frame, text="Exit Program", command=self.exit_program)
        self.exit_btn.pack(fill="x", padx=10, pady=10, side="bottom") # Placed at the bottom of the control panel frame


        # Initialize bbox_predictor based on initial_bbox_model_type
        self._initialize_bbox_predictor()
        # Start the camera capture thread initially
        self.running = True
        self.capture_thread = None
        self._start_camera_capture() # Call internal method to start capture

    def _setup_gui_logger(self):
        """Sets up a custom logging handler to direct INFO/WARNING/ERROR messages to the GUI textbox."""
        class GUILogHandler(logging.Handler):
            def __init__(self, textbox):
                super().__init__()
                self.textbox = textbox
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            def emit(self, record):
                msg = self.format(record)
                # Use after to update GUI from a non-GUI thread safely
                self.textbox.after(0, lambda: self.textbox.insert(ctk.END, msg + "\n"))
                self.textbox.after(0, lambda: self.textbox.see(ctk.END)) # Auto-scroll to the end

        # Remove existing stream handlers to prevent duplicate console output
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logging.root.removeHandler(handler)

        gui_handler = GUILogHandler(self.results_text)
        gui_handler.setLevel(logging.INFO) # Only log INFO and above to GUI
        logging.getLogger().addHandler(gui_handler)

    def _update_ip_cam_url_entry_state(self):
        """Updates the state of the IP Cam URL entry based on selected camera type."""
        if self.camera_type_var.get() == "IP Cam":
            self.ip_cam_url_entry.configure(state="normal")
        else:
            self.ip_cam_url_entry.configure(state="disabled")

    def on_camera_type_selected(self, choice):
        """Callback for when a new camera type is selected from the OptionMenu."""
        self.active_camera_type = choice
        logging.info(f"Camera type selected: {self.active_camera_type}")
        self._update_ip_cam_url_entry_state()
        self.status_label.configure(text=f"Please click 'Apply Camera Settings' to switch to {self.active_camera_type}.", text_color="blue")

    def apply_camera_settings(self):
        """Applies the selected camera settings (type and URL/index) and restarts capture."""
        new_ip_cam_url = self.ip_cam_url_entry.get().strip()
        if self.active_camera_type == "IP Cam" and not new_ip_cam_url:
            messagebox.showerror("Configuration Error", "IP Cam URL cannot be empty when selected.")
            self.status_label.configure(text="Error: IP Cam URL is empty.", text_color="red")
            # Revert to previous camera type if invalid input
            self.camera_type_var.set("Built-in Cam" if self.active_camera_type == "IP Cam" else "IP Cam")
            self.active_camera_type = self.camera_type_var.get()
            self._update_ip_cam_url_entry_state()
            return

        self.ip_cam_url = new_ip_cam_url
        logging.info(f"Applying camera settings: type={self.active_camera_type}, URL={self.ip_cam_url}")
        
        # Stop current capture first
        self._stop_camera_capture()
        # Start new capture with updated settings
        self._start_camera_capture()

    def on_bbox_model_type_selected(self, choice):
        """Callback for when a new bounding box model type is selected."""
        self.active_bbox_model_type = choice
        logging.info(f"BBox model type selected: {self.active_bbox_model_type}")
        self.status_label.configure(text=f"Please click 'Apply Model Settings' to switch to {self.active_bbox_model_type}.", text_color="blue")
    
    def apply_bbox_model_settings(self):
        """Applies the selected bounding box model settings and re-initializes the predictor."""
        logging.info(f"Applying BBox model settings: type={self.active_bbox_model_type}")
        self._initialize_bbox_predictor()
        # After re-initializing, it's good to provide feedback to the user
        if self.bbox_predictor:
            self.status_label.configure(text=f"{self.active_bbox_model_type} applied successfully.", text_color="green")
        else:
            self.status_label.configure(text=f"Failed to apply {self.active_bbox_model_type}.", text_color="red")


    def _initialize_bbox_predictor(self):
        """Initializes the bbox_predictor based on the active_bbox_model_type."""
        if self.active_bbox_model_type == "YuNet Detector":
            try:
                logging.info("Initializing YuNetFaceDetector...")
                self.bbox_predictor = YuNetFaceDetector(
                    model_path=self.yunet_model_path, 
                    input_size=(320, 320), # Default input size for YuNet
                    conf_threshold=0.9, 
                    nms_threshold=0.3, 
                    top_k=5000
                )
                logging.info("YuNetFaceDetector initialized.")
            except Exception as e:
                logging.error(f"Error initializing YuNetFaceDetector: {e}", exc_info=True)
                messagebox.showerror("Model Error", f"Failed to load YuNet Detector: {e}. Check model path and ensure 'models/bbox_models/YuNet/face_detection_yunet_2023mar.onnx' exists.")
                self.status_label.configure(text="Error loading YuNet Detector.", text_color="red")
                self.bbox_predictor = None # Ensure predictor is None on failure
        elif self.active_bbox_model_type == "Haar Cascade Detector":
            try:
                logging.info("Initializing BoundingBoxDetector (Haar Cascade)...")
                # BoundingBoxDetector's __init__ defaults to 'haarcascade_frontalface_default.xml'
                self.bbox_predictor = BoundingBoxDetector(cascade_path=self.haar_cascade_path)
                logging.info("BoundingBoxDetector (Haar Cascade) initialized.")
            except Exception as e:
                logging.error(f"Error initializing BoundingBoxDetector (Haar Cascade): {e}", exc_info=True)
                messagebox.showerror("Model Error", f"Failed to load Haar Cascade Detector: {e}. Check cascade path and ensure '{self.haar_cascade_path}' exists.")
                self.status_label.configure(text="Error loading Haar Cascade Detector.", text_color="red")
                self.bbox_predictor = None # Ensure predictor is None on failure
        elif self.active_bbox_model_type == "Custom CNN Detector":
            try:
                logging.info("Initializing Custom CNN BoundingBoxPredictor...")
                # Assuming BoundingBoxPredictor is the custom CNN model class
                self.bbox_predictor = BoundingBoxPredictor(model_path=self.custom_cnn_model_path, device='cpu')
                logging.info("Custom CNN BoundingBoxPredictor initialized.")
            except Exception as e:
                logging.error(f"Error initializing Custom CNN BoundingBoxPredictor: {e}", exc_info=True)
                messagebox.showerror("Model Error", f"Failed to load Custom CNN Detector: {e}. Check model path and ensure '{self.custom_cnn_model_path}' exists.")
                self.status_label.configure(text="Error loading Custom CNN Detector.", text_color="red")
                self.bbox_predictor = None # Ensure predictor is None on failure
        else:
            logging.error(f"Unknown BBox model type: {self.active_bbox_model_type}")
            messagebox.showerror("Model Error", f"Unknown BBox model type: {self.active_bbox_model_type}")
            self.status_label.configure(text="Unknown BBox model type.", text_color="red")
            self.bbox_predictor = None # Ensure predictor is None

    def update_threshold(self, value):
        """Updates the recognition threshold based on slider movement."""
        self.threshold = float(value)
        self.threshold_label.configure(text=f"Current Threshold: {self.threshold:.5f}")

    def extract_face(self, frame):
        """
        Detects faces in the frame, extracts the first one, and generates its embedding.
        Adds the embedding to a queue for averaging.
        """
        if self.bbox_predictor is None:
            logging.warning("No BBox predictor initialized. Cannot extract face.")
            self.root.after(0, lambda: self.status_label.configure(text="No face detector loaded. Select a model.", text_color="orange"))
            return None

        faces = self.bbox_predictor.predict_bounding_box(frame)
        if len(faces) > 0:
            self.latest_bbox = faces[0] # Store bounding box for drawing
            cropped_face = self.bbox_predictor.crop_face(frame, faces[0])
            embedding = self.embedding_predictor.generate_embedding(cropped_face).flatten()

            with self.lock: # Ensure thread-safe access to the queue
                if self.embedding_queue.full():
                    self.embedding_queue.get() # Remove oldest if queue is full
                self.embedding_queue.put(embedding) # Add new embedding

            return embedding
        self.latest_bbox = None # No face detected
        return None

    def capture_multiple_embeddings(self, num_samples=10):
        """
        Collects and averages recent embeddings from the queue for robust recognition.
        Provides user feedback via the status label.
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        self.status_label.configure(text="Detecting face and collecting samples...", text_color="yellow")
        self.root.update_idletasks() # Force GUI update to show status immediately

        # Clear the queue at the beginning of a new capture operation to ensure fresh samples
        with self.lock:
            while not self.embedding_queue.empty():
                self.embedding_queue.get() # Clear any stale embeddings

        # Initial check: wait briefly for a face to be detected at all
        initial_check_time = time.time()
        face_detected_initially = False
        while time.time() - initial_check_time < 2: # Wait up to 2 seconds for initial detection
            # Check if self.latest_bbox is being updated by the capture_frames_loop
            # This is a proxy for whether a face is being detected
            if self.latest_bbox is not None:
                face_detected_initially = True
                break
            time.sleep(0.05) # Small delay to avoid busy-waiting

        if not face_detected_initially:
            self.status_label.configure(text="No face detected in initial scan.", text_color="red")
            return None

        # Wait until we have enough samples in the queue
        timeout = 5  # seconds to wait for samples (after initial detection)
        start_time = time.time()
        while self.embedding_queue.qsize() < num_samples:
            if time.time() - start_time > timeout:
                if self.embedding_queue.qsize() == 0:
                    self.status_label.configure(text="Failed to collect any samples after initial detection.", text_color="red")
                    return None
                # If we have *some* samples but not `num_samples` due to timeout, take what we have
                self.status_label.configure(text=f"Timeout: Collected {self.embedding_queue.qsize()} of {num_samples} samples.", text_color="orange")
                break # Exit if timeout, but still process existing samples
            time.sleep(0.1) # Small delay while waiting for more samples
        
        # Get all available embeddings from queue and average them
        embeddings = []
        with self.lock:
            # Drain the queue to get all collected embeddings
            while not self.embedding_queue.empty():
                embeddings.append(self.embedding_queue.get())
        
        if embeddings:
            self.status_label.configure(text="Samples collected successfully.", text_color="green")
            return np.mean(embeddings, axis=0) # Return the average embedding
        
        # This case should ideally not be reached if face_detected_initially was True,
        # but added for robustness.
        self.status_label.configure(text="No samples collected (unexpected error).", text_color="red")
        return None
        
    def _start_camera_capture(self):
        """Initializes the active camera source and starts the frame capture thread."""
        self._stop_camera_capture() # Ensure any previous camera is stopped

        if self.active_camera_type == "Built-in Cam":
            self.cap = cv2.VideoCapture(self.built_in_camera_index)
            if not self.cap.isOpened():
                logging.error(f"Failed to open built-in camera with index {self.built_in_camera_index}")
                messagebox.showerror("Camera Error", f"Could not open built-in camera {self.built_in_camera_index}. Please ensure it's connected and not in use.")
                self.status_label.configure(text="Failed to open built-in camera.", text_color="red")
                self.running = False
                return
            logging.info(f"Started built-in camera capture (index: {self.built_in_camera_index}).")
            self.status_label.configure(text="Built-in camera connected.", text_color="green")
        elif self.active_camera_type == "IP Cam":
            if not self.ip_cam_url:
                messagebox.showerror("Configuration Error", "IP Cam URL is not set.")
                self.status_label.configure(text="IP Cam URL not set.", text_color="red")
                self.running = False
                return
            logging.info(f"Attempting to connect to IP Cam at URL: {self.ip_cam_url}")
            self.status_label.configure(text=f"Connecting to IP Cam: {self.ip_cam_url}...", text_color="yellow")
            # IP cam doesn't use cv2.VideoCapture directly, so self.cap remains None
            # The capture_frames method will handle the urllib.request connection.
        else:
            messagebox.showerror("Camera Error", "Unknown camera type selected.")
            self.status_label.configure(text="Unknown camera type selected.", text_color="red")
            self.running = False
            return

        # Start the capture thread only if camera initialization was successful (or for ESP32, if URL is set)
        if self.running: # Check self.running as it might have been set to False by error above
            self.capture_thread = threading.Thread(target=self._capture_frames_loop)
            self.capture_thread.daemon = True # Allows thread to exit when main app exits
            self.capture_thread.start()
            self.root.after(30, self._update_frame_on_gui) # Start GUI frame updates

    def _stop_camera_capture(self):
        """Stops the current camera capture thread and releases resources."""
        if self.capture_thread and self.capture_thread.is_alive():
            self.running = False # Signal the thread to stop
            logging.info("Signaling capture thread to stop.")
            self.capture_thread.join(timeout=1.0) # Wait for the thread to finish
            if self.capture_thread.is_alive():
                logging.warning("Capture thread did not terminate cleanly within timeout.")
        
        if self.cap and self.cap.isOpened():
            logging.info("Releasing OpenCV camera resources.")
            self.cap.release()
            self.cap = None # Reset cap object
        
        # Reset running flag for next start (for _start_camera_capture)
        self.running = True 
        logging.info("Camera capture stopped.")

    def _capture_frames_loop(self):
        """
        Continuously captures frames from the active camera source (built-in or ESP32)
        and processes them. Runs in a separate thread.
        """
        while self.running:
            frame = None
            if self.active_camera_type == "Built-in Cam":
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.error("Failed to grab frame from built-in camera. Attempting to re-acquire...")
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.built_in_camera_index)
                        if not self.cap.isOpened():
                            logging.error("Failed to re-acquire built-in camera. Stopping capture.")
                            # Use root.after to safely interact with GUI from another thread
                            self.root.after(0, lambda: messagebox.showerror("Camera Error", "Built-in camera disconnected or inaccessible. Please restart the application."))
                            self.root.after(0, lambda: self.status_label.configure(text="Built-in camera lost.", text_color="red"))
                            self.running = False # Stop the main capture loop
                        continue # Skip current frame processing
                else:
                    # Camera was not opened or lost, attempt to restart or exit
                    logging.error("Built-in camera object not initialized or closed unexpectedly.")
                    self.running = False
                    self.root.after(0, lambda: messagebox.showerror("Camera Error", "Built-in camera stream lost. Please try restarting the application."))
                    self.root.after(0, lambda: self.status_label.configure(text="Built-in camera stream lost.", text_color="red"))
                    continue

            elif self.active_camera_type == "IP Cam":
                try:
                    img_resp = urllib.request.urlopen(self.ip_cam_url, timeout=0.5) # Shorter timeout for faster feedback
                    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                    if frame is None:
                        raise ValueError("Could not decode image from URL.")
                    # Update status once connection is successful
                    self.root.after(0, lambda: self.status_label.configure(text="IP Cam connected.", text_color="green"))

                except urllib.error.URLError as e:
                    logging.warning(f"IP Cam connection error: {e}. Retrying...")
                    self.root.after(0, lambda: self.status_label.configure(text=f"IP Cam error: {e}. Retrying...", text_color="orange"))
                    time.sleep(1) # Wait before retry
                    continue
                except Exception as e:
                    logging.error(f"Error fetching/decoding IP Cam frame: {e}. Retrying...")
                    self.root.after(0, lambda: self.status_label.configure(text=f"IP Cam decode error: {e}. Retrying...", text_color="orange"))
                    time.sleep(1) # Wait before retry
                    continue
            
            if frame is None:
                # This can happen if camera failed to open/re-open or URL fetch failed
                time.sleep(0.1) # Prevent busy loop if no frame
                continue 

            try:
                # Process the frame (face extraction, bounding box, add to queue)
                self.extract_face(frame) 

                # Draw bounding box on the frame for display
                if self.latest_bbox is not None:
                    x1, y1, x2, y2 = self.latest_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle

                # Convert frame to RGB for CustomTkinter display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = Image.fromarray(rgb_frame)

            except Exception as e:
                logging.error(f"Frame processing error: {e}", exc_info=True) # Log full traceback
                # Continue capturing frames even if an error occurs in processing a single frame
            
            time.sleep(0.01) # Small delay to yield CPU, adjust for desired FPS

        # Ensure camera resources are released when the thread exits
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _update_frame_on_gui(self):
        """Updates the video feed label in the GUI. Called repeatedly by root.after()."""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            # Dynamically resize frame to fit current video_label size for better responsiveness
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            # Only proceed if the label widget has been rendered and has a usable size
            if label_width > 1 and label_height > 1: # Check for minimal size
                img_width, img_height = self.current_frame.size

                # Only proceed if the image itself has valid dimensions
                if img_width > 0 and img_height > 0:
                    img_aspect_ratio = float(img_width) / img_height
                    label_aspect_ratio = float(label_width) / label_height

                    if label_aspect_ratio > img_aspect_ratio:
                        # Label is wider or same height relative to image aspect ratio, fit to label's height
                        new_height = label_height
                        new_width = int(new_height * img_aspect_ratio)
                    else:
                        # Label is taller or same width relative to image aspect ratio, fit to label's width
                        new_width = label_width
                        new_height = int(new_width / img_aspect_ratio)

                    # Ensure calculated dimensions are at least 1 pixel
                    new_width = max(1, new_width)
                    new_height = max(1, new_height)

                    try:
                        # Use Image.Resampling.LANCZOS for high-quality downsampling
                        resized_frame = self.current_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        # Wrap the PIL Image in CTkImage for proper scaling on HighDPI displays
                        self.photo = ctk.CTkImage(light_image=resized_frame, dark_image=resized_frame, size=(new_width, new_height))

                        self.video_label.configure(image=self.photo)
                        self.video_label.image = self.photo # Keep a reference!
                    except ValueError as e:
                        logging.error(f"Error during image resize: {e}. Original: {img_width}x{img_height}, Target: {new_width}x{new_height}")
                else:
                    logging.warning(f"Skipping frame update: current_frame has invalid dimensions {img_width}x{img_height}")
            # else: # Label not ready or too small, skip update for this cycle
                # logging.debug(f"Skipping frame update: video_label not ready or too small {label_width}x{label_height}")

        if self.running: 
            self.root.after(30, self._update_frame_on_gui) # Schedule next frame update (approx 33 FPS)

    def register_person(self):
        """Initiates the registration process in a separate thread."""
        if self.processing:
            messagebox.showwarning("Processing", "Please wait, a registration or login operation is already in progress.")
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Input Error", "Please enter a name for the person before registering.")
            return

        def register_thread_target():
            """Worker function for the registration thread."""
            try:
                self.processing = True
                self._set_interaction_buttons_state("disabled")
                self.status_label.configure(text="Registering: Capturing face samples...", text_color="yellow")
                self.root.update_idletasks() # Force GUI update to show status

                embedding = self.capture_multiple_embeddings(num_samples=self.num_samples)
                if embedding is None:
                    messagebox.showerror("Registration Failed", "Could not extract face features. Please ensure your face is well-lit, clearly visible, and held still within the video frame during capture.")
                    self.status_label.configure(text="Registration failed: No adequate face data.", text_color="red")
                    return

                person_id = self.person_db.add_person(
                    embedding, name, self.is_allowed_var.get()
                )
                self.results_text.insert(ctk.END, f"Registered: {name} (ID: {person_id})\n") # Use ctk.END for appending
                self.name_entry.delete(0, ctk.END) # Clear name entry after successful registration
                self.status_label.configure(text=f"Registered '{name}' successfully!", text_color="green")
                self.root.after(0, self._load_persons_to_gui) # Refresh person list after registration
            except Exception as e:
                logging.error(f"Registration failed for {name}: {e}", exc_info=True) # Log full traceback
                messagebox.showerror("Registration Error", f"An unexpected error occurred during registration: {e}")
                self.status_label.configure(text="Registration failed due to an error.", text_color="red")
            finally:
                self.processing = False
                self._set_interaction_buttons_state("normal")

        threading.Thread(target=register_thread_target).start()
        
    def login_person(self):        
        """Initiates the login (recognition) process in a separate thread."""
        if self.processing:
            messagebox.showwarning("Processing", "Please wait, a registration or login operation is already in progress.")
            return

        def login_thread_target():
            """Worker function for the login thread."""
            try:
                self.processing = True
                self._set_interaction_buttons_state("disabled")
                self.status_label.configure(text="Logging in: Capturing face samples...", text_color="yellow")
                self.root.update_idletasks() # Force GUI update to show status

                embedding = self.capture_multiple_embeddings(num_samples=self.num_samples)
                if embedding is None:
                    messagebox.showerror("Login Failed", "Could not extract face features. Please ensure your face is well-lit, clearly visible, and held still within the video frame during capture.")
                    self.status_label.configure(text="Login failed: No adequate face data.", text_color="red")
                    return

                person_id, person_data, min_distance = self.person_db.find_match(embedding, self.threshold)

                # Update the displayed minimum distance
                self.min_distance_label.configure(text=f"Last Match Distance: {min_distance:.5f}")

                if person_data:
                    access_status = 'Allowed' if person_data['is_allowed'] else 'Denied'
                    status_color = "green" if person_data['is_allowed'] else "red"
                    result = (
                        f"Login Result:\n"
                        f"Name: {person_data['name']}\n"
                        f"Access: {access_status}\n"
                        f"Match Distance: {min_distance:.5f} (Threshold: {self.threshold:.3f})\n" # Added distance and threshold for context
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'-'*30}\n"
                    )
                    self.status_label.configure(text=f"Login: {person_data['name']} - {access_status}!", text_color=status_color)
                else:
                    # No match found within the threshold.
                    result = (
                        f"Login Result:\n"
                        f"No match found.\n"
                        f"Closest Distance: {min_distance:.5f} (Threshold: {self.threshold:.3f})\n" # Show closest distance for context
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'-'*30}\n"
                    )
                    self.status_label.configure(text="Login failed: No match found.", text_color="red")

                self.results_text.insert(ctk.END, result) # Use ctk.END for appending
            except Exception as e:
                logging.error(f"Login failed: {e}", exc_info=True) # Log full traceback
                messagebox.showerror("Login Error", f"An unexpected error occurred during login: {e}")
                self.status_label.configure(text="Login failed due to an error.", text_color="red")
            finally:
                self.processing = False
                self._set_interaction_buttons_state("normal")

        threading.Thread(target=login_thread_target).start()

    def _on_inner_frame_configure(self, event=None):
        """Updates the scrollregion of the canvas when the inner frame changes size."""
        self.person_list_canvas.configure(scrollregion=self.person_list_canvas.bbox("all"))
        # Adjust the width of the canvas window to be the width of the inner frame.
        # This prevents horizontal scrollbar if content fits, and enables it if it doesn't.
        canvas_width = self.person_list_canvas.winfo_width()
        inner_frame_width = self.person_list_inner_frame.winfo_reqwidth()
        if inner_frame_width > canvas_width:
             self.person_list_canvas.itemconfig("inner_frame", width=inner_frame_width)
        else:
             self.person_list_canvas.itemconfig("inner_frame", width=canvas_width)

    def _on_mousewheel_canvas(self, event):
        """Handle mouse wheel scrolling for the canvas (Windows/macOS)."""
        if self.person_list_canvas.winfo_exists(): # Check if widget exists
            self.person_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_mousewheel_canvas_linux(self, event):
        """Handle mouse wheel scrolling for Linux systems for the canvas."""
        if self.person_list_canvas.winfo_exists(): # Check if widget exists
            if event.num == 4: # Scroll up
                self.person_list_canvas.yview_scroll(-1, "units")
            elif event.num == 5: # Scroll down
                self.person_list_canvas.yview_scroll(1, "units")

    def _load_persons_to_gui(self):
        """Loads all registered persons from DB and displays them in the 'Manage Persons' tab."""
        # Clear previous list
        for frame in self.person_row_frames:
            frame.destroy()
        self.person_row_frames = []

        persons = self.person_db.get_all_persons()
        
        # Clear any existing labels in the inner frame
        for widget in self.person_list_inner_frame.winfo_children():
            widget.destroy()

        # Get the text color for labels from the CustomTkinter theme
        current_mode = ctk.get_appearance_mode()
        mode_index = 0 if current_mode == "Light" else 1
        # This should be a light color for dark mode
        label_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"][
            mode_index
        ]

        if not persons:
            label = ctk.CTkLabel(self.person_list_inner_frame, text="No registered persons yet.", font=("Arial", 12), text_color=label_text_color)
            label.grid(row=0, column=0, padx=5, pady=2, sticky="w", columnspan=5)
            # Ensure the inner frame is configured for the single label
            self.person_list_inner_frame.grid_columnconfigure(0, weight=1)
            # Manually trigger configure event to update scrollregion
            self.person_list_inner_frame.event_generate("<Configure>")
            return

        # Configure columns for the inner frame (which holds the grid)
        self.person_list_inner_frame.grid_columnconfigure(0, weight=1, minsize=100) # ID
        self.person_list_inner_frame.grid_columnconfigure(1, weight=2, minsize=150) # Name
        self.person_list_inner_frame.grid_columnconfigure(2, weight=1, minsize=80)  # Allowed
        self.person_list_inner_frame.grid_columnconfigure(3, weight=2, minsize=180) # Registered At
        self.person_list_inner_frame.grid_columnconfigure(4, weight=2, minsize=120) # Delete button

        for i, person in enumerate(persons):
            row_frame = ctk.CTkFrame(self.person_list_inner_frame, fg_color="transparent")
            row_frame.grid(row=i, column=0, columnspan=5, sticky="ew", padx=2, pady=1)
            # Propagate column configurations to the row_frame as well
            row_frame.grid_columnconfigure(0, weight=1, minsize=100) # ID
            row_frame.grid_columnconfigure(1, weight=2, minsize=150) # Name
            row_frame.grid_columnconfigure(2, weight=1, minsize=80)  # Allowed
            row_frame.grid_columnconfigure(3, weight=2, minsize=180) # Registered At
            row_frame.grid_columnconfigure(4, weight=2, minsize=120) # Delete button

            person_id = person['id']
            allowed_status = "Allowed" if person['is_allowed'] else "Denied"
            created_at_formatted = datetime.fromisoformat(person['created_at']).strftime('%Y-%m-%d %H:%M')

            ctk.CTkLabel(row_frame, text=person_id, font=("Arial", 12), text_color=label_text_color).grid(row=0, column=0, padx=2, pady=1, sticky="w")
            ctk.CTkLabel(row_frame, text=person['name'], font=("Arial", 12), text_color=label_text_color).grid(row=0, column=1, padx=2, pady=1, sticky="w")
            ctk.CTkLabel(row_frame, text=allowed_status, font=("Arial", 12), text_color=label_text_color).grid(row=0, column=2, padx=2, pady=1, sticky="w")
            ctk.CTkLabel(row_frame, text=created_at_formatted, font=("Arial", 12), text_color=label_text_color).grid(row=0, column=3, padx=2, pady=1, sticky="w")
            
            delete_btn = ctk.CTkButton(row_frame, text="Delete", 
                                       command=lambda p_id=person_id: self._handle_delete_person_from_list(p_id),
                                       width=80, height=25, fg_color="red", hover_color="darkred")
            delete_btn.grid(row=0, column=4, padx=2, pady=1, sticky="ew")

            self.person_row_frames.append(row_frame)
        
        # After populating, manually trigger configure event to update scrollregion
        self.person_list_inner_frame.event_generate("<Configure>")

        logging.info("Person list refreshed.")

    def _handle_delete_person_from_list(self, person_id):
        """Helper method to handle deletion from the person list directly."""
        if self.processing:
            messagebox.showwarning("Processing", "Please wait, a registration or login operation is already in progress.")
            return

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete person with ID:\n{person_id}?"):
            if self.person_db.delete_person(person_id):
                messagebox.showinfo("Success", f"Person with ID '{person_id}' deleted successfully.")
                logging.info(f"Deleted person with ID: {person_id}")
                self.status_label.configure(text=f"Deleted person '{person_id}' successfully.", text_color="green")
                self._load_persons_to_gui() # Refresh list
            else:
                messagebox.showerror("Error", f"Person with ID '{person_id}' not found in database.")
                logging.warning(f"Attempted to delete non-existent person ID: {person_id}")
                self.status_label.configure(text=f"Error: Person '{person_id}' not found.", text_color="red")

    def _set_interaction_buttons_state(self, state):
        """Helper to enable/disable main interaction buttons and camera/model settings."""
        self.register_btn.configure(state=state)
        self.login_btn.configure(state=state)
        self.apply_camera_settings_btn.configure(state=state)
        self.camera_type_optionmenu.configure(state=state)
        # Ensure IP Cam URL entry state is correct
        self.ip_cam_url_entry.configure(state=state if self.camera_type_var.get() == "IP Cam" else "disabled")
        
        self.apply_bbox_model_btn.configure(state=state)
        self.bbox_model_type_optionmenu.configure(state=state)
        
        # Manage persons tab buttons (only refresh button should be disabled during main ops)
        self.refresh_persons_btn.configure(state=state)
        # Direct delete buttons within the list should also be disabled during main operations
        for frame in self.person_row_frames:
            for child in frame.winfo_children():
                if isinstance(child, ctk.CTkButton):
                    child.configure(state=state)


    def exit_program(self):
        """Handles the clean shutdown of the application."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit the program?"):
            self.running = False
            self.status_label.configure(text="Exiting...", text_color="grey")
            self.root.update_idletasks()

            # Ensure capture thread finishes cleanly
            if self.capture_thread and self.capture_thread.is_alive():
                logging.info("Attempting to join capture thread.")
                self.capture_thread.join(timeout=2.0)
                if self.capture_thread.is_alive():
                    logging.warning("Capture thread did not terminate cleanly. It might be stuck.")
            
            # Release camera resources if they were opened
            if self.cap and self.cap.isOpened():
                logging.info("Releasing camera resources.")
                self.cap.release() 
            
            logging.info("Application shutting down.")
            self.root.quit()
            self.root.destroy()

    def run(self):
        """Starts the main CustomTkinter event loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
        self.root.mainloop()


if __name__ == "__main__":
    database_path = "face_database.db"
    
    # Initialize the embedding predictor (face recognition model)
    logging.info("Loading embedding predictor model...")
    # CONSIDERATION: For a more robust application, add a loading indicator or splash screen here
    # as model loading can take a few seconds and the GUI will be unresponsive during this time.
    embedding_predictor = EmbeddingPredictor(model_path='models/resarksgd/resarksgdaug94.pth', device='cpu')
    logging.info("Embedding predictor loaded.")
    
    # Paths for bounding box models
    yunet_model_path = "models/bbox_models/YuNet/face_detection_yunet_2023mar.onnx"
    # The default Haar cascade path provided by OpenCV (needs opencv-python-headless or opencv-python with cascades)
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # Ensure this path is correct on your system, or pass a specific path if different.
    # e.g., haar_cascade_path = "path/to/haarcascade_frontalface_default.xml"
    custom_cnn_model_path = "models/bbox_models/v5/bbox_v5_randomly_augmented_epoch_3.pth"
    # This path is for the BoundingBoxPredictor class from bounding_box.py


    app = FaceRecognitionApp(
        database_path=database_path,
        embedding_predictor=embedding_predictor,
        initial_camera_index=0, # Default built-in webcam index
        initial_ip_cam_url="http://192.168.1.5/cam-hi.jpg", # Default IP Cam URL (e.g., for an ESP32-CAM)
        initial_camera_type="Built-in Cam", # Default camera source on startup ["Built-in Cam", "IP Cam"]
        initial_bbox_model_type="YuNet Detector", # Default bounding box model on startup
        yunet_model_path=yunet_model_path,
        haar_cascade_path=haar_cascade_path,
        custom_cnn_model_path=custom_cnn_model_path,
        threshold=0.25, # Cosine distance threshold. Lower values require higher similarity (e.g., 0.1 is very strict, 0.4 is more lenient).
        num_samples=5 # Number of samples to average for each embedding (improves accuracy)
    )
    app.run()
