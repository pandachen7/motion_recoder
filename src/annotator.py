import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QSlider, QInputDialog
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint
import cv2
import numpy as np

class Annotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image/Video Annotator")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        controls_layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image/Video/Stream")
        self.load_button.clicked.connect(self.load_source)
        controls_layout.addWidget(self.load_button)

        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size_label)
        controls_layout.addWidget(self.brush_size_slider)

        self.brush_size_label = QLabel(f"Brush Size: {self.brush_size_slider.value()}")
        controls_layout.addWidget(self.brush_size_label)

        self.fill_button = QPushButton("Fill")
        self.fill_button.clicked.connect(self.fill_mask)
        controls_layout.addWidget(self.fill_button)

        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        controls_layout.addWidget(self.save_mask_button)

        self.draw_quad_button = QPushButton("Draw Quadrilateral")
        self.draw_quad_button.setCheckable(True)
        self.draw_quad_button.toggled.connect(self.toggle_draw_quad_mode)
        controls_layout.addWidget(self.draw_quad_button)

        self.layout.addLayout(controls_layout)

        self.source_path = None
        self.original_frame = None
        self.display_frame = None
        self.scale_factor = 1.0

        self.mask = None
        self.drawing = False
        self.last_point = QPoint()
        self.draw_quad_mode = False
        self.quad_points = []


    def toggle_draw_quad_mode(self, checked):
        self.draw_quad_mode = checked
        if not checked:
            self.quad_points = [] # Reset points when mode is turned off
            self.update_display() # Remove any drawn quad

    def update_brush_size_label(self):
        self.brush_size_label.setText(f"Brush Size: {self.brush_size_slider.value()}")

    def fill_mask(self):
        if self.mask is not None:
            cv2.floodFill(self.mask, None, (0, 0), 0)
            self.update_display()

    def save_mask(self):
        if self.mask is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Mask File", "", "PNG Files (*.png)", options=options)
            if fileName:
                # Invert mask so drawing is black and background is white
                inverted_mask = cv2.bitwise_not(self.mask)
                cv2.imwrite(fileName, inverted_mask)

    def load_source(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("All Files (*);;Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi)")

        if file_dialog.exec():
            self.source_path = file_dialog.selectedFiles()[0]
            self.process_source()
        else:
            text, ok = QInputDialog.getText(self, 'RTSP Stream', 'Enter RTSP URL:')
            if ok and text:
                self.source_path = text
                self.process_source()

    def process_source(self):
        if self.source_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.original_frame = cv2.imread(self.source_path)
        else:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                print("Error: Could not open video source.")
                return
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Error: Could not read frame.")
                return
            self.original_frame = frame

        if self.original_frame is None:
            print("Error: Could not load image.")
            return

        self.mask = np.ones((self.original_frame.shape[0], self.original_frame.shape[1]), dtype=np.uint8) * 255
        self.resize_frame()
        self.update_display()

    def resize_frame(self):
        if self.original_frame is None:
            return

        h, w, _ = self.original_frame.shape
        screen_h = self.height() - 150 # Adjust for buttons and controls
        screen_w = self.width() - 20

        if h > 0 and w > 0:
            scale_h = screen_h / h
            scale_w = screen_w / w
            self.scale_factor = min(scale_h, scale_w)

            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)

            self.display_frame = cv2.resize(self.original_frame, (new_w, new_h))

    def update_display(self):
        if self.display_frame is None:
            return

        # Make a copy to avoid issues with QImage buffer
        display_frame_copy = self.display_frame.copy()

        # Resize mask and apply it
        if self.mask is not None:
            display_mask = cv2.resize(self.mask, (display_frame_copy.shape[1], display_frame_copy.shape[0]))
            display_frame_copy[display_mask == 0] = [0, 0, 0] # Black for masked area

        if self.draw_quad_mode and self.quad_points:
            # This part is tricky because QPainter works on QPixmap/QImage, not numpy array directly.
            # We will draw on the pixmap after converting the numpy array.
            pass

        h, w, ch = display_frame_copy.shape
        bytes_per_line = ch * w
        q_img = QImage(display_frame_copy.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        if self.draw_quad_mode and self.quad_points:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 2) # Red pen
            painter.setPen(pen)
            for point in self.quad_points:
                painter.drawEllipse(point, 3, 3)
            if len(self.quad_points) == 4:
                 for i in range(4):
                    painter.drawLine(self.quad_points[i], self.quad_points[(i + 1) % 4])
            painter.end()

        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        self.resize_frame()
        self.update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.display_frame is not None:
            if self.draw_quad_mode:
                label_pos = self.image_label.pos()
                point = event.pos() - label_pos
                self.quad_points.append(point)
                if len(self.quad_points) <= 4:
                    self.update_display()
                if len(self.quad_points) == 4:
                    self.draw_quad_and_area()
            else:
                self.drawing = True
                self.last_point = event.pos()

    def draw_quad_and_area(self):
        if len(self.quad_points) < 4:
            return

        pixmap = self.image_label.pixmap().copy()
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0), 2) # Red pen
        painter.setPen(pen)

        for i in range(4):
            painter.drawLine(self.quad_points[i], self.quad_points[(i + 1) % 4])

        # Calculate area
        scaled_points = np.array([(p.x() / self.scale_factor, p.y() / self.scale_factor) for p in self.quad_points], dtype=np.float32)
        area = cv2.contourArea(scaled_points)

        # Display area
        area_text = f"Area: {area:.2f}"
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        # Find a good position for the text
        text_pos = self.quad_points[0]
        painter.drawText(text_pos.x(), text_pos.y() - 10, area_text)

        self.image_label.setPixmap(pixmap)


    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.MouseButton.LeftButton and self.drawing and self.display_frame is not None:
            painter = QPainter(self.image_label.pixmap())
            pen = QPen(QColor(0,0,0), self.brush_size_slider.value(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            # Adjust for image label position
            label_pos = self.image_label.pos()
            current_point = event.pos() - label_pos
            last_point = self.last_point - label_pos

            painter.drawLine(last_point, current_point)
            self.last_point = event.pos()

            # Update the mask by drawing on a temporary pixmap and then onto the mask
            if self.image_label.pixmap():
                pixmap = self.image_label.pixmap()
                painter = QPainter(pixmap)
                pen = QPen(QColor(0,0,0), self.brush_size_slider.value(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)

                label_pos = self.image_label.pos()
                current_point = event.pos() - label_pos
                last_point = self.last_point - label_pos
                painter.drawLine(last_point, current_point)
                self.last_point = event.pos()
                self.image_label.setPixmap(pixmap)

                # Update the actual mask
                scaled_last_point_x = int(last_point.x() / self.scale_factor)
                scaled_last_point_y = int(last_point.y() / self.scale_factor)
                scaled_current_point_x = int(current_point.x() / self.scale_factor)
                scaled_current_point_y = int(current_point.y() / self.scale_factor)

                cv2.line(self.mask, (scaled_last_point_x, scaled_last_point_y), (scaled_current_point_x, scaled_current_point_y), 0, int(self.brush_size_slider.value() / self.scale_factor))


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.update_display()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = Annotator()
    annotator.show()
    sys.exit(app.exec())
