from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .io import YoloBox


@dataclass(slots=True)
class PixelBox:
    x: int
    y: int
    width: int
    height: int


class ImageCanvas(QWidget):
    box_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._display_rect = QRect()
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._box: PixelBox | None = None
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._box = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_box(self, box: PixelBox | None) -> None:
        self._box = box
        self.update()

    def clear_box(self) -> None:
        self._box = None
        self.update()

    def get_box(self) -> PixelBox | None:
        return self._box

    def current_pixmap_size(self) -> tuple[int, int] | None:
        if self._pixmap is None:
            return None
        return self._pixmap.width(), self._pixmap.height()

    def pixel_box_to_yolo(self) -> YoloBox | None:
        if self._pixmap is None or self._box is None:
            return None
        width = self._pixmap.width()
        height = self._pixmap.height()
        if width <= 0 or height <= 0 or self._box.width <= 0 or self._box.height <= 0:
            return None

        x_center = (self._box.x + self._box.width / 2.0) / width
        y_center = (self._box.y + self._box.height / 2.0) / height
        box_w = self._box.width / width
        box_h = self._box.height / height

        return YoloBox(
            x_center=max(0.0, min(1.0, x_center)),
            y_center=max(0.0, min(1.0, y_center)),
            width=max(0.0, min(1.0, box_w)),
            height=max(0.0, min(1.0, box_h)),
        )

    def yolo_to_pixel_box(self, yolo_box: YoloBox) -> PixelBox | None:
        if self._pixmap is None:
            return None
        width = self._pixmap.width()
        height = self._pixmap.height()
        if width <= 0 or height <= 0:
            return None

        box_w = int(round(yolo_box.width * width))
        box_h = int(round(yolo_box.height * height))
        center_x = yolo_box.x_center * width
        center_y = yolo_box.y_center * height
        x = int(round(center_x - box_w / 2.0))
        y = int(round(center_y - box_h / 2.0))

        clamped = self._clamp_box(x, y, box_w, box_h)
        return clamped

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self._pixmap is None or self._pixmap.isNull():
            return

        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(self._display_rect, scaled)

        if self._box is not None:
            self._draw_box(painter, self._box, QColorName.RED)

        if self._drag_start is not None and self._drag_current is not None:
            preview = self._make_box_from_points(self._drag_start, self._drag_current)
            if preview is not None:
                self._draw_box(painter, preview, QColorName.GREEN)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return
        image_point = self._widget_to_image(event.position().toPoint())
        if image_point is None:
            return
        self._drag_start = image_point
        self._drag_current = image_point
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is None or self._pixmap is None:
            return
        image_point = self._widget_to_image(event.position().toPoint())
        if image_point is None:
            return
        self._drag_current = image_point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._drag_start is None or self._pixmap is None:
            return
        image_point = self._widget_to_image(event.position().toPoint())
        if image_point is None:
            image_point = self._drag_current

        if image_point is not None:
            new_box = self._make_box_from_points(self._drag_start, image_point)
            if new_box is not None and new_box.width > 1 and new_box.height > 1:
                self._box = new_box
                self.box_changed.emit()

        self._drag_start = None
        self._drag_current = None
        self.update()

    def _draw_box(self, painter: QPainter, box: PixelBox, color: Qt.GlobalColor) -> None:
        if self._pixmap is None:
            return
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)

        x_scale = self._display_rect.width() / self._pixmap.width()
        y_scale = self._display_rect.height() / self._pixmap.height()

        x = int(round(self._display_rect.x() + box.x * x_scale))
        y = int(round(self._display_rect.y() + box.y * y_scale))
        w = int(round(box.width * x_scale))
        h = int(round(box.height * y_scale))

        painter.drawRect(x, y, w, h)

    def _widget_to_image(self, point: QPoint) -> QPoint | None:
        if self._pixmap is None or self._display_rect.isNull():
            return None
        if not self._display_rect.contains(point):
            return None

        rel_x = point.x() - self._display_rect.x()
        rel_y = point.y() - self._display_rect.y()
        x_scale = self._pixmap.width() / self._display_rect.width()
        y_scale = self._pixmap.height() / self._display_rect.height()
        image_x = int(rel_x * x_scale)
        image_y = int(rel_y * y_scale)
        return QPoint(image_x, image_y)

    def _make_box_from_points(self, start: QPoint, end: QPoint) -> PixelBox | None:
        x = min(start.x(), end.x())
        y = min(start.y(), end.y())
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        return self._clamp_box(x, y, width, height)

    def _clamp_box(self, x: int, y: int, width: int, height: int) -> PixelBox | None:
        if self._pixmap is None:
            return None
        max_w = self._pixmap.width()
        max_h = self._pixmap.height()

        x = max(0, min(x, max_w - 1))
        y = max(0, min(y, max_h - 1))
        width = max(0, min(width, max_w - x))
        height = max(0, min(height, max_h - y))
        return PixelBox(x=x, y=y, width=width, height=height)


class QColorName:
    RED = Qt.red
    GREEN = Qt.green
