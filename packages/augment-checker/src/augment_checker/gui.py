from __future__ import annotations

import json
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .types import GeometryMetrics, IntegrityIssue, ModelMetrics, SampleRecord
from .yolo import label_to_pixel_corners, load_yolo_labels


class CheckerWindow(QMainWindow):
    def __init__(
        self,
        records: list[SampleRecord],
        integrity_issues: list[IntegrityIssue],
        geometry_metrics: list[GeometryMetrics],
        model_reports: list[ModelMetrics],
    ) -> None:
        super().__init__()
        self.records = records
        self.integrity_issues = integrity_issues
        self.geometry_by_key = {(m.split, m.stem): m for m in geometry_metrics}
        self.model_reports = model_reports

        self.setWindowTitle("Augment Checker")
        self.resize(1300, 850)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)

        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._on_row_changed)

        self.split_filter = QComboBox()
        self.split_filter.addItems(["all", "train", "val"])
        self.split_filter.currentTextChanged.connect(self._reload_list)

        self.outliers_only = QCheckBox("Outliers only")
        self.outliers_only.stateChanged.connect(self._reload_list)

        self.show_green_overlay = QCheckBox("Show green label overlay")
        self.show_green_overlay.setChecked(True)
        self.show_green_overlay.stateChanged.connect(
            lambda _state: self._on_row_changed(self.sample_list.currentRow())
        )

        self.show_red_overlay = QCheckBox("Show red metadata overlay")
        self.show_red_overlay.setChecked(True)
        self.show_red_overlay.stateChanged.connect(
            lambda _state: self._on_row_changed(self.sample_list.currentRow())
        )

        self.model_selector = QComboBox()
        model_names = ["all"] + [m.model_name for m in self.model_reports]
        self.model_selector.addItems(model_names)
        self.model_selector.currentTextChanged.connect(
            lambda _text: self._on_row_changed(self.sample_list.currentRow())
        )

        self.info = QTextEdit()
        self.info.setReadOnly(True)

        self.filtered_records: list[SampleRecord] = []
        self._build_layout()
        self._reload_list()

    def _build_layout(self) -> None:
        side = QVBoxLayout()
        side.addWidget(QLabel("Split"))
        side.addWidget(self.split_filter)
        side.addWidget(self.outliers_only)
        side.addWidget(self.show_green_overlay)
        side.addWidget(self.show_red_overlay)
        side.addWidget(QLabel("Model"))
        side.addWidget(self.model_selector)
        side.addWidget(QLabel("Samples"))
        side.addWidget(self.sample_list, 1)
        side.addWidget(QLabel("Metrics"))
        side.addWidget(self.info, 1)

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(420)

        root = QHBoxLayout()
        root.addWidget(self.image_label, 1)
        root.addWidget(side_widget)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _reload_list(self) -> None:
        split_sel = self.split_filter.currentText()
        outlier_only = self.outliers_only.isChecked()

        filtered: list[SampleRecord] = []
        for rec in self.records:
            if split_sel != "all" and rec.split != split_sel:
                continue
            geom = self.geometry_by_key.get((rec.split, rec.stem))
            if outlier_only and not (geom and geom.is_outlier):
                continue
            filtered.append(rec)

        self.filtered_records = filtered
        self.sample_list.clear()
        for rec in self.filtered_records:
            self.sample_list.addItem(f"{rec.split}/{rec.stem}")

        if self.filtered_records:
            self.sample_list.setCurrentRow(0)

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.filtered_records):
            return
        rec = self.filtered_records[row]
        self._render_image(rec)
        self._render_info(rec)

    def _render_image(self, rec: SampleRecord) -> None:
        if rec.image_path is None:
            return
        img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        if img is None:
            return

        if self.show_green_overlay.isChecked() and rec.label_path is not None:
            try:
                h, w = img.shape[:2]
                labels = load_yolo_labels(rec.label_path, is_prediction=False)
                for label in labels:
                    corners = label_to_pixel_corners(label, w, h).reshape((-1, 1, 2)).astype("int32")
                    cv2.polylines(img, [corners], True, (0, 255, 0), 2)
            except Exception:
                pass

        if self.show_red_overlay.isChecked() and rec.meta_path is not None:
            try:
                meta = json.loads(rec.meta_path.read_text(encoding="utf-8"))
                import numpy as np

                targets = meta.get("targets")
                if not isinstance(targets, list):
                    targets = []
                for t in targets:
                    corners = t.get("projected_corners_px_rect_obb", t.get("projected_corners_px", []))
                    if len(corners) == 4:
                        poly = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [poly], True, (0, 0, 255), 2)
            except Exception:
                pass

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def _render_info(self, rec: SampleRecord) -> None:
        lines = [f"sample: {rec.split}/{rec.stem}"]
        selected_model = self.model_selector.currentText()

        rec_issues = [i for i in self.integrity_issues if i.split == rec.split and i.stem == rec.stem]
        if rec_issues:
            lines.append("integrity:")
            for issue in rec_issues:
                lines.append(f"- {issue.code}: {issue.message}")
        else:
            lines.append("integrity: OK")

        geom = self.geometry_by_key.get((rec.split, rec.stem))
        if geom:
            lines.append(f"geometry evaluable: {geom.evaluable}")
            lines.append(f"mean corner err px: {geom.mean_corner_err_px}")
            lines.append(f"max corner err px: {geom.max_corner_err_px}")
            lines.append(f"obb iou meta-vs-label: {geom.obb_iou_meta_vs_label}")
            lines.append(f"outlier: {geom.is_outlier}")
            if geom.message:
                lines.append(f"geometry message: {geom.message}")

        for model in self.model_reports:
            if selected_model != "all" and model.model_name != selected_model:
                continue
            sample_metric = next((s for s in model.samples if s.split == rec.split and s.stem == rec.stem), None)
            if sample_metric is None:
                continue
            lines.append(
                f"model {model.model_name}: iou={sample_metric.iou}, drift_px={sample_metric.center_drift_px}, missed={sample_metric.missed}"
            )

        self.info.setPlainText("\n".join(lines))


def launch_gui(
    records: list[SampleRecord],
    integrity_issues: list[IntegrityIssue],
    geometry_metrics: list[GeometryMetrics],
    model_reports: list[ModelMetrics],
) -> None:
    app = QApplication([])
    window = CheckerWindow(records, integrity_issues, geometry_metrics, model_reports)
    window.show()
    app.exec()
