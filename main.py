import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QListWidget, QListWidgetItem, QFileDialog,
    QGroupBox, QGridLayout, QProgressBar, QStyle, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import traceback

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Football pitch dimensions in meters
PITCH_WIDTH = 68
PITCH_LENGTH = 105
HALF_LENGTH = PITCH_LENGTH / 2

class CoordinateMapper:
    def __init__(self, video_width, video_height):
        self.video_width = video_width
        self.video_height = video_height
        
        # Define center line points
        self.center_x = video_width / 2
        
        # Define perspective transform points relative to center line
        self.video_points = np.float32([
            [self.center_x - 0.4 * video_width, 0.8 * video_height],  # Bottom left
            [self.center_x + 0.4 * video_width, 0.8 * video_height],  # Bottom right
            [self.center_x - 0.4 * video_width, 0.2 * video_height],  # Top left
            [self.center_x + 0.4 * video_width, 0.2 * video_height]   # Top right
        ])
        
        # Updated pitch points centered around halfway line
        self.pitch_points = np.float32([
            [-HALF_LENGTH, 0],             # Left bottom
            [HALF_LENGTH, 0],              # Right bottom
            [-HALF_LENGTH, PITCH_WIDTH],   # Left top
            [HALF_LENGTH, PITCH_WIDTH]     # Right top
        ])
        
        # Calculate the perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(self.video_points, self.pitch_points)

    def video_to_pitch(self, x, y):
        """Convert video coordinates to pitch coordinates relative to center line"""
        # Apply perspective transform
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)
        pitch_x, pitch_y = transformed[0][0]
        
        # Ensure coordinates are within pitch boundaries
        pitch_x = np.clip(pitch_x, -HALF_LENGTH, HALF_LENGTH)
        pitch_y = np.clip(pitch_y, 0, PITCH_WIDTH)
        
        return pitch_x, pitch_y


class PlayerTracker:
    def __init__(self, coordinate_mapper):
        self.coordinate_mapper = coordinate_mapper
        self.players = {}
        self.next_id = 1
        self.max_distance = 5.0  # Maximum distance in meters for player matching

    def update(self, detections):
        current_positions = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Convert video coordinates to centered pitch coordinates
            pitch_x, pitch_y = self.coordinate_mapper.video_to_pitch(center[0], center[1])
            current_positions.append(((pitch_x, pitch_y), (x1, y1, x2, y2)))

        # Update existing players and create new ones
        updated_players = {}
        used_positions = set()

        # Match existing players to new detections
        for player_id, player_info in self.players.items():
            if not current_positions:
                break
                
            last_pos = player_info['positions'][-1]
            best_match = None
            min_dist = self.max_distance

            for i, (new_pos, bbox) in enumerate(current_positions):
                if i in used_positions:
                    continue
                    
                dist = np.sqrt((last_pos[0] - new_pos[0])**2 + (last_pos[1] - new_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = (i, new_pos, bbox)

            if best_match:
                idx, new_pos, bbox = best_match
                used_positions.add(idx)
                player_info['positions'].append(new_pos)
                player_info['bbox'] = bbox
                updated_players[player_id] = player_info

        # Create new players for unmatched detections
        for i, (pos, bbox) in enumerate(current_positions):
            if i not in used_positions:
                player_id = f"Player {self.next_id}"
                self.next_id += 1
                updated_players[player_id] = {
                    'positions': [pos],
                    'bbox': bbox,
                    'team': 'Unknown',
                    'role': 'Unknown'
                }

        self.players = updated_players
        return self.players

        
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, dict)
    update_slider_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.coordinate_mapper = CoordinateMapper(self.video_width, self.video_height)
        self.tracker = PlayerTracker(self.coordinate_mapper)
        self.paused = False
        self.current_frame = 0
        self.selected_player = None
        self.running = True
        self.playback_speed = 1.0
    def run(self):
        while self.cap.isOpened() and self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)
                    self.current_frame += 1
                    self.update_slider_signal.emit(self.current_frame)
                    # Adjust sleep time based on playback speed
                    self.msleep(int(50 / self.playback_speed))
                else:
                    self.status_signal.emit("Video playback complete")
                    break
            else:
                self.msleep(50)

        self.cap.release()

    def process_frame(self, frame):
        results = model(frame)
        detections = results.pandas().xyxy[0].to_numpy()
        
        # Update player tracking 
        players = self.tracker.update(detections)
        
        # Draw on frame
        processed_frame = frame.copy()
        for player_id, info in players.items():
            x1, y1, x2, y2 = info['bbox']
            
            # Different colors for selected player
            color = (0, 255, 0) if player_id == self.selected_player else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and position
            center_x, center_y = info['positions'][-1]
            pitch_x, pitch_y = self.map_to_pitch_coordinates(center_x, center_y)
            
            label = f"{player_id}"
            if info['team'] != 'Unknown':
                label += f" ({info['team']})"
            
            cv2.putText(processed_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.change_pixmap_signal.emit(processed_frame, players)

    def map_to_pitch_coordinates(self, x, y):
        pitch_x = (x / self.video_width) * PITCH_WIDTH
        pitch_y = (y / self.video_height) * PITCH_LENGTH
        return pitch_x, pitch_y

    def set_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)

    def set_selected_player(self, player_id):
        self.selected_player = player_id

    def set_player_team(self, player_id, team):
        if player_id in self.tracker.players:
            self.tracker.players[player_id]['team'] = team

    def set_player_role(self, player_id, role):
        if player_id in self.tracker.players:
            self.tracker.players[player_id]['role'] = role

    def set_playback_speed(self, speed):
        self.playback_speed = speed

    def pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        self.wait()

class FootballTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Football Player Tracker")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize tracking variables
        self.video_path = None
        self.thread = None
        self.current_player = None
        self.reference_player = None  # Initialize reference player
        self.reference_heatmap = None  # Initialize reference heatmap
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)

        # Left panel - Player Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 0, 0, 2, 1)

        # Center panel - Video
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 0, 1, 1, 1)

        # Bottom panel - Controls
        bottom_panel = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel, 1, 1, 1, 1)

        # Right panel - Analysis
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 0, 2, 2, 1)

        # Set stretch factors
        main_layout.setColumnStretch(0, 1)  # Left panel
        main_layout.setColumnStretch(1, 2)  # Center panel
        main_layout.setColumnStretch(2, 1)  # Right panel

        self.setup_style()

    def create_left_panel(self):
        group = QGroupBox("Player Management")
        layout = QVBoxLayout()

        # Player List
        self.player_list_widget = QListWidget()
        self.player_list_widget.itemClicked.connect(self.player_selected)
        layout.addWidget(QLabel("Players:"))
        layout.addWidget(self.player_list_widget)

        # Team Selection
        team_layout = QHBoxLayout()
        team_layout.addWidget(QLabel("Team:"))
        self.team_combo = QComboBox()
        self.team_combo.addItems(['Unknown', 'Team A', 'Team B'])
        self.team_combo.currentTextChanged.connect(self.team_changed)
        team_layout.addWidget(self.team_combo)
        layout.addLayout(team_layout)

        # Role Selection
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role:"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(['Unknown', 'Forward', 'Midfielder', 'Defender', 'Goalkeeper'])
        self.role_combo.currentTextChanged.connect(self.role_changed)
        role_layout.addWidget(self.role_combo)
        layout.addLayout(role_layout)

        group.setLayout(layout)
        return group

    def reset_tracking(self):
        """Reset tracking variables when loading a new video"""
        self.current_player = None
        self.reference_player = None
        self.reference_heatmap = None

    def create_center_panel(self):
        group = QGroupBox("Video Feed")
        layout = QVBoxLayout()

        # Import button
        self.import_button = QPushButton("Import Video")
        self.import_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.import_button.clicked.connect(self.import_video)
        layout.addWidget(self.import_button)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        layout.addWidget(self.video_label)

        group.setLayout(layout)
        return group

    def create_bottom_panel(self):
        group = QGroupBox("Playback Controls")
        layout = QVBoxLayout()

        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_button)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.5x', '1.0x', '1.5x', '2.0x'])
        self.speed_combo.setCurrentText('1.0x')
        self.speed_combo.currentTextChanged.connect(self.speed_changed)
        speed_layout.addWidget(self.speed_combo)
        controls_layout.addLayout(speed_layout)

        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setOrientation(Qt.Orientation.Horizontal)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        group.setLayout(layout)
        return group

    def create_right_panel(self):
        group = QGroupBox("Analysis")
        layout = QVBoxLayout()

        # Heatmap
        layout.addWidget(QLabel("Movement Heatmap:"))
        self.figure = plt.figure(figsize=(6, 8))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Statistics
        self.stats_label = QLabel("Player Statistics:")
        layout.addWidget(self.stats_label)

        group.setLayout(layout)
        return group

    def setup_style(self):
        # Set dark theme
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        self.setPalette(dark_palette)

    def import_video(self):
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Import Video",
                "",
                "Video Files (*.mp4 *.avi *.mkv)"
            )

            if file_name:
                if self.thread is not None:
                    self.thread.stop()
                    self.thread.wait()
                
                self.reset_tracking()  # Reset tracking variables
                self.video_path = file_name
                self.status_label.setText("Loading video...")
                self.start_tracking()

    def start_tracking(self):
        if self.video_path:
            self.thread = VideoThread(self.video_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.update_slider_signal.connect(self.update_progress)
            self.thread.status_signal.connect(self.update_status)
            self.progress_bar.setMaximum(self.thread.total_frames)
            self.thread.start()
            self.status_label.setText("Tracking players...")

    def player_selected(self, item):
        player_id = item.text().split(" (")[0]  # Handle case where team is in the text
        if self.thread:
            if self.current_player != player_id:    
                self.current_player = player_id
                self.thread.set_selected_player(player_id)
                
                # Set reference player if not already set
                if self.reference_player is None:
                    self.reference_player = player_id
                    self.create_reference_heatmap(player_id)
                
                # Update team and role combos
                if player_id in self.thread.tracker.players:
                    player_info = self.thread.tracker.players[player_id]
                    self.team_combo.setCurrentText(player_info['team'])
                    self.role_combo.setCurrentText(player_info['role'])
                self.update_player_stats(player_id)

    def update_heatmap_display(self):
        """Update the heatmap display using the stored reference heatmap"""
        if self.reference_heatmap is not None:
            self.draw_pitch()
            
            # Plot the reference heatmap
            self.ax.imshow(self.reference_heatmap, 
                          extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
                          origin='lower',
                          cmap='hot',
                          alpha=0.6)
            
            # Add center line indicator
            self.ax.axvline(x=HALF_LENGTH, color='white', linestyle='--', alpha=0.5)
            
            # Update title to show this is the reference player
            self.ax.set_title(f"Reference Player: {self.reference_player}", color='white', pad=10)
            self.canvas.draw()


    def create_reference_heatmap(self, player_id):
        """Create and store the reference heatmap for the first selected player"""
        if self.thread and player_id in self.thread.tracker.players:
            positions = self.thread.tracker.players[player_id]['positions']
            
            # Create heatmap data
            heatmap = np.zeros((int(PITCH_WIDTH), int(PITCH_LENGTH)))
            for pitch_x, pitch_y in positions:
                x_idx = int(pitch_x + HALF_LENGTH)
                y_idx = int(pitch_y)
                if 0 <= x_idx < PITCH_LENGTH and 0 <= y_idx < PITCH_WIDTH:
                    heatmap[y_idx, x_idx] += 1

            # Apply Gaussian smoothing
            heatmap = gaussian_filter(heatmap, sigma=2)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            
            self.reference_heatmap = heatmap
            self.update_heatmap_display()



    def team_changed(self, team):
        if self.current_player and self.thread:
            self.thread.set_player_team(self.current_player, team)
            self.update_player_list(self.thread.tracker.players)

    def role_changed(self, role):
        if self.current_player and self.thread:
            self.thread.set_player_role(self.current_player, role)
            self.update_player_list(self.thread.tracker.players)

    def speed_changed(self, speed_text):
        if self.thread:
            speed = float(speed_text.replace('x', ''))
            self.thread.set_playback_speed(speed)

    def update_image(self, cv_img, players):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        self.update_player_list(players)
        
        if self.current_player and self.current_player in players:
            self.update_player_stats(self.current_player)
            # Only update heatmap display if reference player's positions changed
            if self.reference_player == self.current_player:
                self.create_reference_heatmap(self.reference_player)


    def update_player_list(self, players):
        current_item = self.player_list_widget.currentItem()
        current_selected = current_item.text().split(" (")[0] if current_item else None

        self.player_list_widget.clear()
        for player_id, info in players.items():
            display_text = player_id
            if info['team'] != 'Unknown':
                display_text += f" ({info['team']})"
            if info['role'] != 'Unknown':
                display_text += f" - {info['role']}"
                
            item = QListWidgetItem(display_text)
            self.player_list_widget.addItem(item)
            if player_id == current_selected:
                item.setSelected(True)
                self.player_list_widget.setCurrentItem(item)

    def update_player_stats(self, player_id):
        if self.thread and player_id in self.thread.tracker.players:
            player = self.thread.tracker.players[player_id]
            positions = player['positions']
            
            # Calculate statistics using pitch coordinates
            total_distance = 0
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    x1, y1 = positions[i-1]
                    x2, y2 = positions[i]
                    total_distance += np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calculate average position
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            
            stats_text = f"""
            Player: {player_id}
            Team: {player['team']}
            Role: {player['role']}
            Distance: {total_distance:.1f}m
            Avg Position: ({avg_x:.1f}m, {avg_y:.1f}m)
            """
            self.stats_label.setText(stats_text)


    def update_heatmap(self, player_id):
        if self.thread and player_id in self.thread.tracker.players:
            self.draw_pitch()
            positions = self.thread.tracker.players[player_id]['positions']
            
            # Create heatmap data using centered pitch coordinates
            # Convert from [-HALF_LENGTH, HALF_LENGTH] to [0, PITCH_LENGTH] for display
            heatmap = np.zeros((int(PITCH_WIDTH), int(PITCH_LENGTH)))
            for pitch_x, pitch_y in positions:
                # Convert from centered coordinates to array indices
                x_idx = int(pitch_x + HALF_LENGTH)
                y_idx = int(pitch_y)
                if 0 <= x_idx < PITCH_LENGTH and 0 <= y_idx < PITCH_WIDTH:
                    heatmap[y_idx, x_idx] += 1

            # Apply Gaussian smoothing
            heatmap = gaussian_filter(heatmap, sigma=2)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            # Plot heatmap
            self.ax.imshow(heatmap, 
                        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
                        origin='lower',
                        cmap='hot',
                        alpha=0.6)
            
            # Add center line indicator
            self.ax.axvline(x=HALF_LENGTH, color='white', linestyle='--', alpha=0.5)
            
            self.ax.set_title(f"{player_id} Movement Heatmap", color='white', pad=10)
            self.canvas.draw()

    def draw_pitch(self):
        """Draw a horizontal football pitch on the matplotlib axes"""
        self.ax.clear()

        # Set background color to green
        self.ax.set_facecolor('#2e8b57')

        # Draw pitch outline
        self.ax.plot([0, PITCH_LENGTH], [0, 0], 'white', linewidth=1)
        self.ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], 'white', linewidth=1)
        self.ax.plot([0, 0], [0, PITCH_WIDTH], 'white', linewidth=1)
        self.ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], 'white', linewidth=1)

        # Halfway line (center)
        self.ax.plot([HALF_LENGTH, HALF_LENGTH], [0, PITCH_WIDTH], 'white', linewidth=2)

        # Center circle
        center_circle = plt.Circle((HALF_LENGTH, PITCH_WIDTH/2), 9.15, fill=False, color='white', linewidth=1)
        self.ax.add_artist(center_circle)

        # Center dot
        center_dot = plt.Circle((HALF_LENGTH, PITCH_WIDTH/2), 0.5, color='white')
        self.ax.add_artist(center_dot)

        # Penalty areas
        self.ax.plot([0, 16.5], [13.85, 13.85], 'white', linewidth=1)
        self.ax.plot([0, 16.5], [54.15, 54.15], 'white', linewidth=1)
        self.ax.plot([16.5, 16.5], [13.85, 54.15], 'white', linewidth=1)

        self.ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH], [13.85, 13.85], 'white', linewidth=1)
        self.ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH], [54.15, 54.15], 'white', linewidth=1)
        self.ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH-16.5], [13.85, 54.15], 'white', linewidth=1)

        # Goal areas
        self.ax.plot([0, 5.5], [24.85, 24.85], 'white', linewidth=1)
        self.ax.plot([0, 5.5], [43.15, 43.15], 'white', linewidth=1)
        self.ax.plot([5.5, 5.5], [24.85, 43.15], 'white', linewidth=1)

        self.ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH], [24.85, 24.85], 'white', linewidth=1)
        self.ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH], [43.15, 43.15], 'white', linewidth=1)
        self.ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH-5.5], [24.85, 43.15], 'white', linewidth=1)

        # Set axis limits with minimal padding
        self.ax.set_xlim(-1, PITCH_LENGTH + 1)
        self.ax.set_ylim(-1, PITCH_WIDTH + 1)

        # Remove axis labels and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set aspect ratio to equal
        self.ax.set_aspect('equal')

        # Tight layout to remove extra padding
        self.figure.tight_layout()
        
    def toggle_play_pause(self):
        if self.thread:
            self.thread.pause()
            icon = QStyle.StandardPixmap.SP_MediaPlay if self.thread.paused else QStyle.StandardPixmap.SP_MediaPause
            self.play_pause_button.setIcon(self.style().standardIcon(icon))

    def update_progress(self, frame_number):
        self.progress_bar.setValue(frame_number)
        # Calculate and display timestamp
        if self.thread:
            fps = self.thread.cap.get(cv2.CAP_PROP_FPS)
            current_time = frame_number / fps
            total_time = self.thread.total_frames / fps
            self.status_label.setText(f"Time: {int(current_time//60):02d}:{int(current_time%60):02d} / "
                                    f"{int(total_time//60):02d}:{int(total_time%60):02d}")

    def update_status(self, message):
        self.status_label.setText(message)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = convert_to_Qt_format.scaled(800, 450, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(scaled)

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootballTrackerGUI()
    window.show()
    sys.exit(app.exec_())