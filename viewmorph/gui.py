import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from face_landmarks import (
    detect_and_extract_AKAZE,
    detect_and_extract_ORB,
    extract_points,
    get_landmarks,
)
from morphing import (  # assuming these functions are added to morphing.py
    create_morph_frames,
    morph_images,
    save_gif,
)


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.image1 = None
        self.image2 = None
        self.points1 = None
        self.points2 = None
        self.alpha = 0.0

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_morph)
        self.layout.addWidget(self.slider)

        self.alpha_label = QLabel(self)
        self.alpha_label.setText(f"alpha: {self.alpha:.2f}")
        self.layout.addWidget(self.alpha_label)

        self.frame_spinbox = QSpinBox(self)
        self.frame_spinbox.setRange(1, 100)  # adjust the range as needed
        self.frame_spinbox.setValue(10)  # default value
        self.layout.addWidget(self.frame_spinbox)

        self.save_gif_button = QPushButton("Save as GIF", self)
        self.save_gif_button.clicked.connect(self.save_gif)
        self.layout.addWidget(self.save_gif_button)

        # Add radio buttons to select the feature detection algorithm
        self.button_group = QButtonGroup(self)
        self.radio_dlib = QRadioButton("dlib", self)
        self.radio_orb = QRadioButton("ORB", self)
        self.radio_akaze = QRadioButton("AKAZE", self)
        self.radio_dlib.setChecked(True)
        self.button_group.addButton(self.radio_dlib)
        self.button_group.addButton(self.radio_orb)
        self.button_group.addButton(self.radio_akaze)
        self.layout.addWidget(self.radio_dlib)
        self.layout.addWidget(self.radio_orb)
        self.layout.addWidget(self.radio_akaze)

        # Add a horizontal layout for the images
        self.images_layout = QHBoxLayout()
        self.layout.addLayout(self.images_layout)

        # Add labels to display the morphed images
        self.label_dlib = QLabel("dlib", self)
        self.image_label_dlib = QLabel(self)
        self.images_layout.addWidget(self.label_dlib)
        self.images_layout.addWidget(self.image_label_dlib)

        self.label_orb = QLabel("ORB", self)
        self.image_label_orb = QLabel(self)
        self.images_layout.addWidget(self.label_orb)
        self.images_layout.addWidget(self.image_label_orb)

        self.label_akaze = QLabel("AKAZE", self)
        self.image_label_akaze = QLabel(self)
        self.images_layout.addWidget(self.label_akaze)
        self.images_layout.addWidget(self.image_label_akaze)

        # Add label to display error messages
        self.error_label = QLabel(self)
        self.layout.addWidget(self.error_label)

    def update_morph(self, value=None):
        # Morph the images and update the canvas
        if self.image1 is not None and self.image2 is not None:
            self.alpha = self.slider.value() / 100.0

            # Apply each feature detection algorithm
            for alg in ["dlib", "ORB", "AKAZE"]:
                if alg == "dlib":
                    points1 = self.points1_dlib
                    points2 = self.points2_dlib
                elif alg == "ORB":
                    points1 = self.points1_orb
                    points2 = self.points2_orb
                else:  # AKAZE
                    points1 = self.points1_akaze
                    points2 = self.points2_akaze

                if points1.shape != points2.shape:
                    self.error_label.setText(
                        f"Skipping morphing for {alg} due to mismatch in the number of keypoints."
                    )
                    continue

                morphed_image = morph_images(
                    self.image1, self.image2, points1, points2, self.alpha
                )
                morphed_image = cv2.cvtColor(morphed_image, cv2.COLOR_BGR2RGB)
                qimage = QImage(
                    morphed_image,
                    morphed_image.shape[1],
                    morphed_image.shape[0],
                    QImage.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimage)

                if alg == "dlib":
                    self.image_label_dlib.setPixmap(pixmap)
                elif alg == "ORB":
                    self.image_label_orb.setPixmap(pixmap)
                else:  # AKAZE
                    self.image_label_akaze.setPixmap(pixmap)

            self.alpha_label.setText(f"alpha: {self.alpha:.2f}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.xpm *.jpg)"
        )
        if file_path:
            self.image1 = cv2.imread(file_path)

            # Apply each feature detection algorithm
            for alg in ["dlib", "ORB", "AKAZE"]:
                if alg == "dlib":
                    landmarks = get_landmarks(self.image1)
                    points = extract_points(landmarks)
                elif alg == "ORB":
                    points, _ = detect_and_extract_ORB(self.image1)
                else:  # AKAZE
                    points, _ = detect_and_extract_AKAZE(self.image1)

                if alg == "dlib":
                    self.points1_dlib = points
                elif alg == "ORB":
                    self.points1_orb = points
                else:  # AKAZE
                    self.points1_akaze = points

            # Flip the image and get its landmarks
            self.image2 = cv2.flip(self.image1, 1)

            # Apply each feature detection algorithm
            for alg in ["dlib", "ORB", "AKAZE"]:
                if alg == "dlib":
                    landmarks = get_landmarks(self.image2)
                    points = extract_points(landmarks)
                elif alg == "ORB":
                    points, _ = detect_and_extract_ORB(self.image2)
                else:  # AKAZE
                    points, _ = detect_and_extract_AKAZE(self.image2)

                if alg == "dlib":
                    self.points2_dlib = points
                elif alg == "ORB":
                    self.points2_orb = points
                else:  # AKAZE
                    self.points2_akaze = points

            self.update_morph()

    def save_gif(self):
        # Create the morph frames and save as GIF
        if self.image1 is not None and self.image2 is not None:
            if self.radio_dlib.isChecked():
                points1 = self.points1_dlib
                points2 = self.points2_dlib
            elif self.radio_orb.isChecked():
                points1 = self.points1_orb
                points2 = self.points2_orb
            else:  # AKAZE
                points1 = self.points1_akaze
                points2 = self.points2_akaze

            frames = create_morph_frames(
                self.image1, self.image2, points1, points2, self.frame_spinbox.value()
            )
            gif_path = QFileDialog.getSaveFileName(
                self, "Save as", "", "GIF Files (*.gif)"
            )[0]
            if gif_path:
                save_gif(frames, gif_path)


app = QApplication(sys.argv)
ex = App()
ex.show()
sys.exit(app.exec_())
