"""版式分析融合模块。

将 Layout（PP-DocLayoutV3）、YOLO、Cell 三个模型的检测结果融合，
输出标准化的 layout_analysis.json 格式，供 OCR 阶段使用。

标准类别：
    title        标题
    category_no  分类号（右上角手写数字，如 3-2）
    page_no      页码（左下角或右下角）
    table        表格整体框
    table_cell   表格单元格框
    text         普通文本框
    photo        照片
    seal         印章
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class LayoutFusionConfig:
    """融合参数配置。"""

    # 类别映射：Layout 原始类别 -> 标准类别
    layout_class_map: dict[str, str] = field(default_factory=lambda: {
        "doc_title": "title",
        "paragraph_title": "title",
        "figure_title": "title",
        "header": "category_no",
        "table": "table",
        "text": "text",
        "footer": "text",
        "seal": "seal",
        "photo": "photo",
    })

    # YOLO 原始类别 -> 标准类别（YOLO 已训练的类别直接映射）
    yolo_class_map: dict[str, str] = field(default_factory=lambda: {
        "title": "title",
        "class number": "category_no",
        "page number": "page_no",
        "photo": "photo",
        "seal": "seal",
    })

    # number 类别消歧：根据位置判断属于 category_no 还是 page_no
    # 右上角区域阈值（相对图像尺寸）
    number_top_right_x_ratio: float = 0.5   # x > width * ratio → 右半部分
    number_top_right_y_ratio: float = 0.3   # y < height * ratio → 上部
    # 底部区域阈值
    number_bottom_y_ratio: float = 0.7      # y > height * ratio → 底部

    # IoU 去重阈值：同一类别 IoU 超过此值视为重复
    iou_threshold: float = 0.5

    # 是否同时输出表格整体框（Cell 检测到单元格时）
    keep_table_whole_box: bool = True


# ---------------------------------------------------------------------------
# 融合模块
# ---------------------------------------------------------------------------

class LayoutFusion:
    """版式分析融合器。

    用法::

        config = LayoutFusionConfig()
        fusion = LayoutFusion(config)
        result = fusion.fuse(
            layout_boxes=layout_boxes,
            yolo_results=yolo_results,
            table_cell_payload=table_cell_payload,
            image_size=(width, height),
        )
    """

    def __init__(self, config: LayoutFusionConfig | None = None) -> None:
        self._config = config or LayoutFusionConfig()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fuse(
        self,
        layout_boxes: list[dict[str, Any]],
        yolo_results: Any = None,
        table_cell_payload: dict[str, Any] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """融合三个模型的输出，返回标准化 Box 列表。

        Args:
            layout_boxes: Layout 模型的原始 boxes（含 label, score, coordinate）。
            yolo_results: YOLO 模型的原始 predict 结果（ultralytics 格式）。
            table_cell_payload: Cell 模型的 detect_cells 返回值。
            image_size: (width, height)，用于 number 消歧。

        Returns:
            标准化的 Box 字典列表，每个包含 Number, Class, Position, score, source。
        """
        # 1. 收集所有检测结果，按标准类别分组
        all_detections: dict[str, list[dict[str, Any]]] = {}

        # 1a. 处理 Layout 结果
        layout_items = self._process_layout_boxes(layout_boxes, image_size)
        for item in layout_items:
            cls = item["Class"]
            all_detections.setdefault(cls, []).append(item)

        # 1b. 处理 YOLO 结果
        yolo_items = self._process_yolo_results(yolo_results)
        for item in yolo_items:
            cls = item["Class"]
            all_detections.setdefault(cls, []).append(item)

        # 2. 两个模型结果都保留，不做去重

        # 3. 处理表格：用 Cell 结果细化
        if table_cell_payload and "table" in all_detections:
            table_boxes_std = all_detections["table"]
            table_whole, table_cells = self._merge_table_cells(
                table_boxes_std, table_cell_payload
            )
            all_detections["table"] = table_whole
            if table_cells:
                all_detections.setdefault("table_cell", []).extend(table_cells)

        # 4. 组装输出，分配 Number
        output_boxes: list[dict[str, Any]] = []
        # 按固定顺序输出各类别
        category_order = [
            "title", "category_no", "page_no",
            "table", "table_cell", "text",
            "photo", "seal",
        ]
        for cls in category_order:
            for box in all_detections.get(cls, []):
                box["Number"] = f"{cls}_{len([b for b in output_boxes if b['Class'] == cls])}"
                output_boxes.append(box)

        # 处理未在 category_order 中的类别
        for cls, items in all_detections.items():
            if cls not in category_order:
                for box in items:
                    box["Number"] = f"{cls}_{len([b for b in output_boxes if b['Class'] == cls])}"
                    output_boxes.append(box)

        return output_boxes

    # ------------------------------------------------------------------
    # Layout 处理
    # ------------------------------------------------------------------

    def _process_layout_boxes(
        self,
        boxes: list[dict[str, Any]],
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """将 Layout 原始 boxes 转换为标准格式。"""
        items: list[dict[str, Any]] = []
        for box in boxes:
            raw_label = str(box.get("label", ""))
            score = float(box.get("score", 0.0))
            coordinate = box.get("coordinate")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
                continue

            # 映射类别
            mapped_class = self._config.layout_class_map.get(raw_label)

            # number 类别消歧
            if raw_label == "number" and image_size is not None:
                mapped_class = self._disambiguate_number(coordinate, image_size)
            elif raw_label == "number":
                mapped_class = "number"  # 无法消歧时保留

            if mapped_class is None:
                # 未映射的类别保留原名
                mapped_class = raw_label

            position = self._to_position_quad(coordinate)
            if position is None:
                continue

            items.append({
                "Class": mapped_class,
                "Position": position,
                "score": score,
                "source": "layout",
            })

        return items

    def _disambiguate_number(
        self,
        coordinate: list[float],
        image_size: tuple[int, int],
    ) -> str:
        """根据位置判断 layout 检测到的 number 是 category_no 还是 page_no。"""
        x_min, y_min, x_max, y_max = coordinate
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width, height = image_size

        cfg = self._config

        # 右上角 → category_no
        if x_center > width * cfg.number_top_right_x_ratio and y_center < height * cfg.number_top_right_y_ratio:
            return "category_no"

        # 底部 → page_no
        if y_center > height * cfg.number_bottom_y_ratio:
            return "page_no"

        # 无法确定，保留为 number
        return "number"

    # ------------------------------------------------------------------
    # YOLO 处理
    # ------------------------------------------------------------------

    def _process_yolo_results(self, yolo_results: Any) -> list[dict[str, Any]]:
        """将 YOLO 的 ultralytics 推理结果转换为标准格式。"""
        if yolo_results is None:
            return []

        items: list[dict[str, Any]] = []
        try:
            if not yolo_results or len(yolo_results) == 0:
                return []

            result = yolo_results[0]
            names = getattr(yolo_results, 'names', None) or getattr(result, 'names', None)
            if names is None:
                return []

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name_raw = names.get(cls_id, str(cls_id)).lower() if isinstance(names, dict) else str(cls_id)
                score = float(box.conf[0])

                # 映射到标准类别
                mapped_class = self._config.yolo_class_map.get(cls_name_raw)
                if mapped_class is None:
                    # 未映射的 YOLO 类别保留原名，不丢弃
                    mapped_class = cls_name_raw

                xyxy = box.xyxy[0].tolist()
                if len(xyxy) != 4:
                    continue

                position = self._to_position_quad(xyxy)
                if position is None:
                    continue

                items.append({
                    "Class": mapped_class,
                    "Position": position,
                    "score": score,
                    "source": "yolo",
                })

        except Exception:
            pass

        return items

    # ------------------------------------------------------------------
    # 表格单元格合并
    # ------------------------------------------------------------------

    def _merge_table_cells(
        self,
        table_boxes_std: list[dict[str, Any]],
        table_cell_payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将 Cell 检测结果与 Layout 的 table 框合并。

        Returns:
            (table_whole_boxes, table_cell_boxes)
            table_whole_boxes: 表格整体框列表（如果 keep_table_whole_box=True）
            table_cell_boxes: 单元格框列表，含 parent_table 关联
        """
        tables = table_cell_payload.get("tables", [])
        if not isinstance(tables, list):
            return table_boxes_std, []

        whole_boxes: list[dict[str, Any]] = []
        cell_boxes: list[dict[str, Any]] = []

        for table_idx, table_info in enumerate(tables):
            if not isinstance(table_info, dict):
                continue

            table_id = f"t{table_idx}"

            # 表格整体框：使用 Cell 报告的 table_coordinate 或 Layout 的框
            table_coord = table_info.get("table_coordinate")
            if table_coord and len(table_coord) == 4:
                position = self._to_position_quad(table_coord)
                if position:
                    whole_boxes.append({
                        "Class": "table",
                        "Position": position,
                        "score": float(table_info.get("score", 0.0) or 0.0),
                        "source": "cell",
                        "_table_id": table_id,
                    })

            # 单元格
            cells = table_info.get("cells", [])
            if not isinstance(cells, list):
                continue

            for cell_idx, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    continue
                coord = cell.get("coordinate")
                if not isinstance(coord, (list, tuple)) or len(coord) != 4:
                    continue

                position = self._to_position_quad(coord)
                if position is None:
                    continue

                cell_boxes.append({
                    "Class": "table_cell",
                    "Position": position,
                    "score": float(cell.get("score", 0.0) or 0.0),
                    "source": "cell",
                    "parent_table": table_id,
                })

        # 如果 Cell 没有返回有效结果，保留原始 Layout table 框
        if not whole_boxes:
            whole_boxes = table_boxes_std

        # 决定是否输出表格整体框
        if self._config.keep_table_whole_box:
            return whole_boxes, cell_boxes
        else:
            # 不输出整体框时，如果没有单元格则保留整体框
            if cell_boxes:
                return [], cell_boxes
            return whole_boxes, []

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------

    def _deduplicate(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对同类别检测结果按 IoU 去重，保留置信度更高的。"""
        if len(items) <= 1:
            return items

        # 按 score 降序排列
        sorted_items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)
        kept: list[dict[str, Any]] = []

        for item in sorted_items:
            is_duplicate = False
            for existing in kept:
                iou = self._compute_iou(item["Position"], existing["Position"])
                if iou > self._config.iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(item)

        return kept

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def _to_position_quad(coordinate: list[float] | tuple[float, ...]) -> list[float] | None:
        """将 [x1, y1, x2, y2] 转换为八点坐标 [左下, 右下, 右上, 左上]。

        即 [x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]。
        """
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
            return None
        try:
            x_min, y_min, x_max, y_max = [float(v) for v in coordinate]
        except (TypeError, ValueError):
            return None
        if x_max <= x_min or y_max <= y_min:
            return None
        return [x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]

    @staticmethod
    def _compute_iou(pos_a: list[float], pos_b: list[float]) -> float:
        """计算两个八点坐标框的 IoU。"""
        # 从八点坐标提取 [x_min, y_min, x_max, y_max]
        def _bbox_from_quad(pos: list[float]) -> tuple[float, float, float, float] | None:
            if len(pos) < 8:
                return None
            xs = [pos[i] for i in range(0, 8, 2)]
            ys = [pos[i] for i in range(1, 8, 2)]
            return min(xs), min(ys), max(xs), max(ys)

        box_a = _bbox_from_quad(pos_a)
        box_b = _bbox_from_quad(pos_b)
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # 交集
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union
