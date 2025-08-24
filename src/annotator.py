import sys

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QSlider,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from loglo import getUniqueLogger

log = getUniqueLogger(__file__)


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
            QIcon.fromTheme("document-open"), "Open File", self
        )
        self.load_file_action.triggered.connect(self.load_file)
        self.toolbar.addAction(self.load_file_action)
        self.load_file_action.setToolTip("Open File")

        self.load_rtsp_action = QAction(
            QIcon.fromTheme("network-wired"), "Open RTSP", self
        )
        self.load_rtsp_action.triggered.connect(self.load_rtsp)
        self.toolbar.addAction(self.load_rtsp_action)
        self.load_rtsp_action.setToolTip("Open RTSP")

        self.toolbar.addSeparator()

        self.brush_action = QAction(QIcon.fromTheme("draw-freehand"), "Brush", self)
        self.brush_action.setCheckable(True)
        self.brush_action.setChecked(True)
        self.toolbar.addAction(self.brush_action)
        self.brush_action.setToolTip("Brush")

        self.erase_action = QAction(QIcon.fromTheme("draw-eraser"), "Erase", self)
        self.erase_action.setCheckable(True)
        self.toolbar.addAction(self.erase_action)
        self.erase_action.setToolTip("Erase")

        self.fill_action = QAction(QIcon.fromTheme("color-fill"), "Fill", self)
        self.fill_action.setCheckable(True)
        self.fill_action.toggled.connect(self.toggle_fill_mode)
        self.toolbar.addAction(self.fill_action)
        self.fill_action.setToolTip("Fill")

        self.draw_quad_action = QAction(
            QIcon.fromTheme("draw-polygon"), "Draw Quad", self
        )
        self.draw_quad_action.setCheckable(True)
        self.draw_quad_action.toggled.connect(self.toggle_draw_quad_mode)
        self.toolbar.addAction(self.draw_quad_action)
        self.draw_quad_action.setToolTip("Draw Quad")

        self.draw_bbox_action = QAction(QIcon.fromTheme("draw-rectangle"), "", self)
        self.draw_bbox_action.setCheckable(True)
        self.brush_action.toggled.connect(self.toggle_brush_mode)
        self.erase_action.toggled.connect(self.toggle_erase_mode)
        self.draw_bbox_action.toggled.connect(self.toggle_draw_bbox_mode)
        self.toolbar.addAction(self.draw_bbox_action)
        self.draw_bbox_action.setToolTip("Draw Bbox")

        self.toolbar.addSeparator()

        self.save_mask_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save Mask",
            self,
        )
        self.save_mask_action.triggered.connect(self.save_mask)
        self.toolbar.addAction(self.save_mask_action)

        # self.save_annotations_action = QAction(
        #     style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        #     "Save Annotations",
        #     self,
        # )
        # self.save_annotations_action.triggered.connect(self.save_annotations)
        # self.toolbar.addAction(self.save_annotations_action)

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
        self.fill_mode = False

        # for test
        self.source_path = "./img/mask.png"
        self.process_source()

    def toggle_brush_mode(self, checked):
        if checked:
            self.erase_action.setChecked(False)
            self.draw_quad_action.setChecked(False)
            self.draw_bbox_action.setChecked(False)
            self.fill_action.setChecked(False)

    def toggle_erase_mode(self, checked):
        if checked:
            self.brush_action.setChecked(False)
            self.draw_quad_action.setChecked(False)
            self.draw_bbox_action.setChecked(False)
            self.fill_action.setChecked(False)

    def toggle_draw_bbox_mode(self, checked):
        self.draw_bbox_mode = checked
        if checked:
            self.draw_quad_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.erase_action.setChecked(False)
            self.fill_action.setChecked(False)

    def toggle_draw_quad_mode(self, checked):
        self.draw_quad_mode = checked
        if checked:
            self.draw_bbox_mode = False
            self.brush_action.setChecked(False)
            self.erase_action.setChecked(False)
            self.fill_action.setChecked(False)
        if not checked:
            self.quad_points = []  # Reset points when mode is turned off
            self.update_display()  # Remove any drawn quad

    def toggle_fill_mode(self, checked):
        self.fill_mode = checked
        if checked:
            self.brush_action.setChecked(False)
            self.erase_action.setChecked(False)
            self.draw_quad_action.setChecked(False)
            self.draw_bbox_action.setChecked(False)
            self.draw_area_action.setChecked(False)

    def update_brush_size_label(self):
        self.brush_size_label.setText(f"Brush: {self.brush_size_slider.value()}")

    def fill_mask_at_point(self, point):
        if self.mask is not None:
            h, w = self.mask.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            seed_point = (point.x(), point.y())
            if seed_point[0] < 0 or seed_point[0] >= w or seed_point[1] < 0 or seed_point[1] >= h:
                log.w(f"Fill point {seed_point} is outside of mask dimensions {(w, h)}")
                return
            # We fill with black (0)
            cv2.floodFill(self.mask, mask, seed_point, 0)
            self.update_display()

    def save_mask(self):
        if self.mask is not None:
            fileName, _ = QFileDialog.getSaveFileName(
                self,
                "Save Mask File",
                "",
                "PNG Files (*.png)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if fileName:
                # Invert mask so drawing is black and background is white
                inverted_mask = cv2.bitwise_not(self.mask)
                cv2.imwrite(fileName, inverted_mask)

    # def save_annotations(self):
    #     if not self.bboxes and self.mask is None:
    #         return

    #     fileName, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Save Annotations",
    #         "",
    #         "All Files (*)",
    #         options=QFileDialog.DontUseNativeDialog,
    #     )

    #     if fileName:
    #         if self.bboxes:
    #             with open(f"{fileName}_bbox.txt", "w") as f:
    #                 for bbox in self.bboxes:
    #                     p1 = bbox[0] - self.image_label.pos()
    #                     p2 = bbox[1] - self.image_label.pos()
    #                     x1 = p1.x() / self.scale_factor
    #                     y1 = p1.y() / self.scale_factor
    #                     x2 = p2.x() / self.scale_factor
    #                     y2 = p2.y() / self.scale_factor
    #                     f.write(f"{x1},{y1},{x2},{y2}\n")
    #         if self.mask is not None:
    #             inverted_mask = cv2.bitwise_not(self.mask)
    #             cv2.imwrite(f"{fileName}_mask.png", inverted_mask)

    def load_file(self):
        options = QFileDialog.Option.DontUseNativeDialog

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
        self._process_source()

    def _process_source(self):
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
        self._resize_frame()

    def _resize_frame(self):
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
        self._update_display()

    def _update_display(self):
        if self.display_frame is None:
            return

        # Make a copy to avoid issues with QImage buffer
        display_frame_copy = self.display_frame.copy()

        # Create base pixmap from the frame
        h, w, ch = display_frame_copy.shape
        bytes_per_line = ch * w
        q_img = QImage(
            display_frame_copy.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img)
        log.d(f"pixmap size: {pixmap.size()}")

        # Start a single painter for all drawing operations
        painter = QPainter(pixmap)

        # Draw the mask overlay
        if self.mask is not None:
            display_mask = cv2.resize(self.mask, (pixmap.width(), pixmap.height()))

            # Create a 4-channel RGBA numpy array for the overlay
            rgba_mask = np.zeros((pixmap.height(), pixmap.width(), 4), dtype=np.uint8)
            rgba_mask[display_mask == 0] = [0, 0, 0, 150]  # semi-transparent black

            # Convert to QImage and then QPixmap
            h_mask, w_mask, ch_mask = rgba_mask.shape
            bytes_per_line_mask = ch_mask * w_mask
            mask_q_img = QImage(
                rgba_mask.data,
                w_mask,
                h_mask,
                bytes_per_line_mask,
                QImage.Format.Format_ARGB32,
            )
            mask_pixmap = QPixmap.fromImage(mask_q_img)

            painter.drawPixmap(0, 0, mask_pixmap)

        # Draw other annotations (quads, bboxes)
        if self.draw_quad_mode and self.quad_points:
            pen = QPen(QColor(255, 0, 0), 2)  # Red pen
            painter.setPen(pen)
            for point in self.quad_points:
                painter.drawEllipse(point, 3, 3)
            if len(self.quad_points) == 4:
                for i in range(4):
                    painter.drawLine(self.quad_points[i], self.quad_points[(i + 1) % 4])

        log.d("t1")  # Keep original log points
        if self.draw_bbox_mode or self.bboxes:
            self.draw_bboxes(painter)

        log.d("t2")
        painter.end()
        self.image_label.setPixmap(pixmap)
        log.d("t3")

    def _map_window_to_image_coords(self, window_pos):
        if self.image_label.pixmap() is None or self.scale_factor == 0:
            return None

        # Top-left corner of the label in the window
        label_rect = self.image_label.geometry()

        # Top-left corner of the pixmap within the label (due to alignment)
        pixmap_size = self.image_label.pixmap().size()
        label_size = self.image_label.size()

        pixmap_x_offset = (label_size.width() - pixmap_size.width()) / 2
        pixmap_y_offset = (label_size.height() - pixmap_size.height()) / 2

        # Mouse position relative to the pixmap
        pixmap_pos = window_pos - label_rect.topLeft() - QPoint(int(pixmap_x_offset), int(pixmap_y_offset))

        # Check if click is inside the pixmap
        if not (0 <= pixmap_pos.x() < pixmap_size.width() and 0 <= pixmap_pos.y() < pixmap_size.height()):
            return None

        # Scale to original image coordinates
        image_x = pixmap_pos.x() / self.scale_factor
        image_y = pixmap_pos.y() / self.scale_factor

        return QPoint(int(image_x), int(image_y))

    def resizeEvent(self, event):
        self.resize_frame()
        self.update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.display_frame is not None
        ):
            if self.fill_mode:
                image_point = self._map_window_to_image_coords(event.pos())
                if image_point:
                    self.fill_mask_at_point(image_point)
                return

            # The other drawing modes still have the coordinate issue,
            # but we focus on the mask drawing first as requested.
            if self.draw_quad_mode:
                label_pos = self.image_label.pos()
                point = event.pos() - label_pos
                self.quad_points.append(point)
                if len(self.quad_points) <= 4:
                    self.update_display()
                if len(self.quad_points) == 4:
                    pixmap = self.image_label.pixmap().copy()
                    self.draw_quad_and_area(pixmap)
            elif self.draw_bbox_mode:
                image_point = self._map_window_to_image_coords(event.pos())
                if image_point:
                    self.drawing = True
                    self.current_bbox = [image_point, image_point]
            else:  # Brush or erase mode
                image_point = self._map_window_to_image_coords(event.pos())
                if image_point:
                    self.drawing = True
                    self.last_point = image_point

    def draw_quad_and_area(self, pixmap):
        if len(self.quad_points) < 4:
            return

        log.d("draw_quad_and_area")
        # pixmap = self.image_label.pixmap().copy()
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
        self._display_quad_area(area, painter, pixmap)
        painter.end()

    def _display_quad_area(self, area, painter, pixmap):
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
        self._mouseMoveEvent(event)

    def _mouseMoveEvent(self, event):
        if (
            event.buttons()
            and Qt.MouseButton.LeftButton
            and self.drawing
            and self.display_frame is not None
        ):
            # The other drawing modes still have the coordinate issue,
            # but we focus on the mask drawing first as requested.
            if self.draw_bbox_mode and self.drawing:
                image_point = self._map_window_to_image_coords(event.pos())
                if image_point:
                    self.current_bbox[1] = image_point
                self.update_display()
            elif self.brush_action.isChecked() or self.erase_action.isChecked():
                current_image_point = self._map_window_to_image_coords(event.pos())
                if current_image_point is None or self.last_point is None:
                    return

                if not self.image_label.pixmap():
                    return

                # Draw on display pixmap for immediate feedback
                pixmap = self.image_label.pixmap()
                painter = QPainter(pixmap)
                color = (
                    QColor(0, 0, 0)
                    if self.brush_action.isChecked()
                    else QColor(255, 255, 255)
                )
                pen = QPen(
                    color,
                    self.brush_size_slider.value(),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(pen)

                # Convert image coords back to scaled pixmap coords for drawing
                last_pixmap_point = self.last_point * self.scale_factor
                current_pixmap_point = current_image_point * self.scale_factor
                painter.drawLine(last_pixmap_point, current_pixmap_point)
                painter.end()
                self.image_label.setPixmap(pixmap)

                # Draw on the actual mask (full resolution)
                mask_color = 0 if self.brush_action.isChecked() else 255
                brush_size_on_mask = max(1, int(self.brush_size_slider.value() / self.scale_factor))
                cv2.line(
                    self.mask,
                    (self.last_point.x(), self.last_point.y()),
                    (current_image_point.x(), current_image_point.y()),
                    mask_color,
                    brush_size_on_mask,
                )

                self.last_point = current_image_point


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_bbox_mode and self.drawing:
                # The final point should also be mapped
                image_point = self._map_window_to_image_coords(event.pos())
                if image_point and self.current_bbox:
                    self.current_bbox[1] = image_point

                # Prevent adding a zero-sized or invalid bbox
                if self.current_bbox and self.current_bbox[0] != self.current_bbox[1]:
                    self.bboxes.append(self.current_bbox)

                self.drawing = False
                self.current_bbox = None
            else:
                self.drawing = False
            self.update_display()

    def draw_bboxes(self, painter):
        # Draw saved bboxes (in green)
        pen = QPen(QColor(0, 255, 0), 2)  # Green pen for saved bboxes
        painter.setPen(pen)
        for bbox_img_coords in self.bboxes:
            # Convert image coordinates to pixmap coordinates for drawing
            p1 = bbox_img_coords[0] * self.scale_factor
            p2 = bbox_img_coords[1] * self.scale_factor
            rect = QRect(p1.toPoint(), p2.toPoint()).normalized()
            painter.drawRect(rect)

        # Draw the bbox currently being drawn (in red)
        if self.current_bbox:
            pen.setColor(QColor(255, 0, 0))  # Red pen for drawing bbox
            painter.setPen(pen)

            # Convert image coordinates to pixmap coordinates for drawing
            p1_pix = self.current_bbox[0] * self.scale_factor
            p2_pix = self.current_bbox[1] * self.scale_factor
            rect = QRect(p1_pix.toPoint(), p2_pix.toPoint()).normalized()
            painter.drawRect(rect)

            # --- This is the new part from the reference snippet ---
            # Calculate size in original image coordinates
            orig_p1 = self.current_bbox[0]
            orig_p2 = self.current_bbox[1]
            w = abs(orig_p2.x() - orig_p1.x())
            h = abs(orig_p2.y() - orig_p1.y())

            # Don't show text for a single point
            if w == 0 and h == 0:
                return

            text = f"{w}x{h}={w * h}"
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(text)
            text_height = font_metrics.height()

            # Position text near the second point of the bbox
            text_pos = p2_pix.toPoint() + QPoint(15, 15)

            # Draw text background
            bg_rect = QRect(
                text_pos,
                QPoint(text_pos.x() + text_width + 4, text_pos.y() + text_height),
            )
            painter.fillRect(bg_rect, QColor(0, 0, 0, 150))

            # Draw text
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(
                text_pos + QPoint(2, text_height - font_metrics.descent()), text
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    annotator = Annotator()
    annotator.show()
    sys.exit(app.exec())
