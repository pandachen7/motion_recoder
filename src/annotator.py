import sys

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class Annotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image/Video Annotator")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # --- Actions ---
        style = self.style()
        self.load_file_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileIcon), "Open File", self
        )
        self.load_file_action.triggered.connect(self.load_file)
        self.toolbar.addAction(self.load_file_action)

        self.load_rtsp_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), "Open RTSP", self
        )
        self.load_rtsp_action.triggered.connect(self.load_rtsp)
        self.toolbar.addAction(self.load_rtsp_action)

        self.toolbar.addSeparator()

        self.brush_action = QAction(QIcon.fromTheme("draw-freehand"), "Brush", self)
        self.brush_action.setCheckable(True)
        self.brush_action.setChecked(True)
        self.toolbar.addAction(self.brush_action)

        self.erase_action = QAction(QIcon.fromTheme("draw-eraser"), "Erase", self)
        self.erase_action.setCheckable(True)
        self.toolbar.addAction(self.erase_action)

        self.fill_action = QAction(QIcon.fromTheme("color-fill"), "Fill", self)
        self.fill_action.triggered.connect(self.fill_mask)
        self.toolbar.addAction(self.fill_action)

        self.draw_quad_action = QAction(
            QIcon.fromTheme("draw-polygon"), "Draw Quad", self
        )
        self.draw_quad_action.setCheckable(True)
        self.draw_quad_action.toggled.connect(self.toggle_draw_quad_mode)
        self.toolbar.addAction(self.draw_quad_action)

        self.draw_bbox_action = QAction(
            QIcon.fromTheme("draw-rectangle"), "Draw Bbox", self
        )
        self.draw_bbox_action.setCheckable(True)
        self.brush_action.toggled.connect(self.toggle_brush_mode)
        self.erase_action.toggled.connect(self.toggle_erase_mode)
        self.draw_bbox_action.toggled.connect(self.toggle_draw_bbox_mode)
        self.toolbar.addAction(self.draw_bbox_action)

        self.toolbar.addSeparator()

        self.save_mask_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save Mask",
            self,
        )
        self.save_mask_action.triggered.connect(self.save_mask)
        self.toolbar.addAction(self.save_mask_action)

        self.save_annotations_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save Annotations",
            self,
        )
        self.save_annotations_action.triggered.connect(self.save_annotations)
        self.toolbar.addAction(self.save_annotations_action)

        # Brush size slider in a container to add to toolbar
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size_label)
        self.brush_size_label = QLabel(f"Brush: {self.brush_size_slider.value()}")
        slider_layout.addWidget(self.brush_size_label)
        slider_layout.addWidget(self.brush_size_slider)
        self.toolbar.addWidget(slider_container)

        self.source_path = None
        self.original_frame = None
        self.display_frame = None
        self.scale_factor = 1.0

        self.mask = None
        self.drawing = False
        self.last_point = QPoint()
        self.draw_quad_mode = False
        self.quad_points = []
        self.draw_bbox_mode = False
        self.bboxes = []
        self.current_bbox = None

    def toggle_brush_mode(self, checked):
        if checked:
            self.erase_action.setChecked(False)
            self.draw_quad_action.setChecked(False)
            self.draw_bbox_action.setChecked(False)

    def toggle_erase_mode(self, checked):
        if checked:
            self.brush_action.setChecked(False)
            self.draw_quad_action.setChecked(False)
            self.draw_bbox_action.setChecked(False)

    def toggle_draw_bbox_mode(self, checked):
        self.draw_bbox_mode = checked
        if checked:
            self.draw_quad_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.erase_action.setChecked(False)

    def toggle_draw_quad_mode(self, checked):
        self.draw_quad_mode = checked
        if checked:
            self.draw_bbox_mode = False
            self.brush_action.setChecked(False)
            self.erase_action.setChecked(False)
        if not checked:
            self.quad_points = []  # Reset points when mode is turned off
            self.update_display()  # Remove any drawn quad

    def update_brush_size_label(self):
        self.brush_size_label.setText(f"Brush: {self.brush_size_slider.value()}")

    def fill_mask(self):
        if self.mask is not None:
            cv2.floodFill(self.mask, None, (0, 0), 0)
            self.update_display()

    def save_mask(self):
        if self.mask is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(
                self, "Save Mask File", "", "PNG Files (*.png)", options=options
            )
            if fileName:
                # Invert mask so drawing is black and background is white
                inverted_mask = cv2.bitwise_not(self.mask)
                cv2.imwrite(fileName, inverted_mask)

    def save_annotations(self):
        if not self.bboxes and self.mask is None:
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotations",
            "",
            "All Files (*)",
            options=options,
        )

        if fileName:
            if self.bboxes:
                with open(f"{fileName}_bbox.txt", "w") as f:
                    for bbox in self.bboxes:
                        p1 = bbox[0] - self.image_label.pos()
                        p2 = bbox[1] - self.image_label.pos()
                        x1 = p1.x() / self.scale_factor
                        y1 = p1.y() / self.scale_factor
                        x2 = p2.x() / self.scale_factor
                        y2 = p2.y() / self.scale_factor
                        f.write(f"{x1},{y1},{x2},{y2}\n")
            if self.mask is not None:
                inverted_mask = cv2.bitwise_not(self.mask)
                cv2.imwrite(f"{fileName}_mask.png", inverted_mask)

    def load_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or Video File",
            "",
            "All Files (*);;Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi)",
            options=options,
        )
        if fileName:
            self.source_path = fileName
            self.process_source()

    def load_rtsp(self):
        text, ok = QInputDialog.getText(self, "RTSP Stream", "Enter RTSP URL:")
        if ok and text:
            self.source_path = text
            self.process_source()

    def process_source(self):
        if self.source_path.lower().endswith((".png", ".jpg", ".jpeg")):
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

        self.mask = (
            np.ones(
                (self.original_frame.shape[0], self.original_frame.shape[1]),
                dtype=np.uint8,
            )
            * 255
        )
        self.resize_frame()
        self.update_display()

    def resize_frame(self):
        if self.original_frame is None:
            return

        h, w, _ = self.original_frame.shape
        screen_h = self.height() - 150  # Adjust for buttons and controls
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
            display_mask = cv2.resize(
                self.mask, (display_frame_copy.shape[1], display_frame_copy.shape[0])
            )
            display_frame_copy[display_mask == 0] = [0, 0, 0]  # Black for masked area

        if self.draw_quad_mode and self.quad_points:
            # This part is tricky because QPainter works on QPixmap/QImage, not numpy array directly.
            # We will draw on the pixmap after converting the numpy array.
            pass

        h, w, ch = display_frame_copy.shape
        bytes_per_line = ch * w
        q_img = QImage(
            display_frame_copy.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img)

        if self.draw_quad_mode and self.quad_points:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 2)  # Red pen
            painter.setPen(pen)
            for point in self.quad_points:
                painter.drawEllipse(point, 3, 3)
            if len(self.quad_points) == 4:
                for i in range(4):
                    painter.drawLine(self.quad_points[i], self.quad_points[(i + 1) % 4])
            painter.end()

        if self.draw_bbox_mode or self.bboxes:
            painter = QPainter(pixmap)
            self.draw_bboxes(painter)
            painter.end()

        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        self.resize_frame()
        self.update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.display_frame is not None
        ):
            if self.draw_quad_mode:
                label_pos = self.image_label.pos()
                point = event.pos() - label_pos
                self.quad_points.append(point)
                if len(self.quad_points) <= 4:
                    self.update_display()
                if len(self.quad_points) == 4:
                    self.draw_quad_and_area()
            elif self.draw_bbox_mode:
                self.drawing = True
                self.current_bbox = [event.pos(), event.pos()]
            else:
                self.drawing = True
                self.last_point = event.pos()

    def draw_quad_and_area(self):
        if len(self.quad_points) < 4:
            return

        pixmap = self.image_label.pixmap().copy()
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0), 2)  # Red pen
        painter.setPen(pen)

        for i in range(4):
            painter.drawLine(self.quad_points[i], self.quad_points[(i + 1) % 4])

        # Calculate area
        scaled_points = np.array(
            [
                (p.x() / self.scale_factor, p.y() / self.scale_factor)
                for p in self.quad_points
            ],
            dtype=np.float32,
        )
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
        if (
            event.buttons()
            and Qt.MouseButton.LeftButton
            and self.drawing
            and self.display_frame is not None
        ):
            if self.draw_bbox_mode:
                self.current_bbox[1] = event.pos()
                self.update_display()
            elif self.brush_action.isChecked() or self.erase_action.isChecked():
                color = 0 if self.brush_action.isChecked() else 255
                painter = QPainter(self.image_label.pixmap())
                pen = QPen(
                    QColor(0, 0, 0) if self.brush_action.isChecked() else QColor(255, 255, 255),
                    self.brush_size_slider.value(),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
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
                    pen = QPen(
                        QColor(0, 0, 0) if self.brush_action.isChecked() else QColor(255, 255, 255),
                        self.brush_size_slider.value(),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                        Qt.PenJoinStyle.RoundJoin,
                    )
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
                    scaled_current_point_x = int(
                        current_point.x() / self.scale_factor
                    )
                    scaled_current_point_y = int(
                        current_point.y() / self.scale_factor
                    )

                    cv2.line(
                        self.mask,
                        (scaled_last_point_x, scaled_last_point_y),
                        (scaled_current_point_x, scaled_current_point_y),
                        color,
                        int(self.brush_size_slider.value() / self.scale_factor),
                    )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_bbox_mode:
                self.drawing = False
                self.bboxes.append(self.current_bbox)
                self.current_bbox = None
            else:
                self.drawing = False
            self.update_display()

    def draw_bboxes(self, painter):
        pen = QPen(QColor(0, 255, 0), 2)  # Green pen for bboxes
        painter.setPen(pen)
        for bbox in self.bboxes:
            p1 = bbox[0] - self.image_label.pos()
            p2 = bbox[1] - self.image_label.pos()
            painter.drawRect(p1.x(), p1.y(), p2.x() - p1.x(), p2.y() - p1.y())
        if self.current_bbox:
            p1 = self.current_bbox[0] - self.image_label.pos()
            p2 = self.current_bbox[1] - self.image_label.pos()
            painter.drawRect(p1.x(), p1.y(), p2.x() - p1.x(), p2.y() - p1.y())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    annotator = Annotator()
    annotator.show()
    sys.exit(app.exec())
