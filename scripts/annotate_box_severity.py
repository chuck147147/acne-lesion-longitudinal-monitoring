"""Local GUI for assigning one project severity grade to each existing box."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "manual_subtype_pilot" / "annotations.csv"

GRADES = (
    ("0*", "0*　仅痘印 / 瘢痕"),
    ("I", "I　粉刺"),
    ("II", "II　丘疹"),
    ("III", "III　脓疱"),
    ("IV", "IV　结节 / 囊肿"),
)
GRADE_NAMES = dict(GRADES)
GRADE_KEYS = {str(index): grade for index, (grade, _) in enumerate(GRADES)}
GRADE_COLORS = {
    "0*": "#858B93",
    "I": "#00B8D9",
    "II": "#FF9F1C",
    "III": "#E63946",
    "IV": "#B5179E",
    "uncertain": "#858B93",
    "unlabeled": "#DDE6ED",
}
GRADE_HELP = (
    "0*：框内只有痘印/瘢痕，没有活动性皮损。",
    "I：开放或闭合粉刺。",
    "II：炎性丘疹；无脓疱、结节/囊肿。",
    "III：脓疱；无结节/囊肿。",
    "IV：结节或囊肿。",
)


def box_coordinates(row: dict[str, str], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    xc = float(row["x_center_norm"]) * image_width
    yc = float(row["y_center_norm"]) * image_height
    width = float(row["width_norm"]) * image_width
    height = float(row["height_norm"]) * image_height
    return xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2


class BoxSeverityAnnotator:
    def __init__(self, root: tk.Tk, csv_path: Path, suggestions_path: Path | None = None) -> None:
        self.root = root
        self.csv_path = csv_path
        self.image_dir = csv_path.parent / "images"
        self.suggestions_by_key: dict[tuple[str, str], dict[str, str]] = {}
        if suggestions_path is not None:
            if not suggestions_path.is_file():
                raise FileNotFoundError(f"模型建议表不存在：{suggestions_path}")
            with suggestions_path.open("r", newline="", encoding="utf-8-sig") as handle:
                for suggestion in csv.DictReader(handle):
                    key = (suggestion.get("image", ""), str(suggestion.get("box_id", "")))
                    self.suggestions_by_key[key] = suggestion
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.headers = list(reader.fieldnames or [])
            self.rows = [dict(row) for row in reader]
        required = {
            "image", "source_filename_grade", "box_id", "x_center_norm", "y_center_norm",
            "width_norm", "height_norm", "box_severity_grade", "review_note",
        }
        missing = required.difference(self.headers)
        if missing:
            raise ValueError(f"标注表缺少列：{', '.join(sorted(missing))}")
        manifest = csv_path.parent / "prediction_summary.csv"
        self.image_names = []
        if manifest.is_file():
            with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
                self.image_names.extend(row["image"] for row in csv.DictReader(handle) if row.get("image"))
        source_manifest = csv_path.parent / "image_manifest.csv"
        self.source_filename_by_image: dict[str, str] = {}
        self.companion_by_image: dict[str, str] = {}
        if source_manifest.is_file():
            with source_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    image_name = (row.get("image") or "").strip()
                    if not image_name:
                        continue
                    if image_name not in self.image_names:
                        self.image_names.append(image_name)
                    self.source_filename_by_image[image_name] = row.get("source_filename", "")
                    self.companion_by_image[image_name] = (row.get("companion_image") or "").strip()
        for row in self.rows:
            if row["image"] not in self.image_names:
                self.image_names.append(row["image"])
        if not self.image_names:
            raise ValueError("标注表和图片清单中都没有图片")
        self.rows_by_image: dict[str, list[int]] = {name: [] for name in self.image_names}
        for index, row in enumerate(self.rows):
            self.rows_by_image.setdefault(row["image"], []).append(index)
        self.image_index = 0
        self.current_image_name = ""
        self.current_rows: list[int] = []
        self.current_row_index: int | None = None
        self.original_image: Image.Image | None = None
        self.display_scale = 1.0
        self.zoom = 1.0
        self.show_ids = True
        self.add_box_mode = False
        self.redraw_box_mode = False
        self.drag_start: tuple[float, float] | None = None
        self.temp_rectangle: int | None = None
        self.deleted_rows: list[tuple[int, dict[str, str]]] = []
        self.next_box_id_by_image = {
            name: max((int(row["box_id"]) for row in self.rows if row["image"] == name), default=0) + 1
            for name in self.image_names
        }
        self.canvas_photo: ImageTk.PhotoImage | None = None
        self.crop_photo: ImageTk.PhotoImage | None = None
        self.companion_photo: ImageTk.PhotoImage | None = None

        first_pending = next((i for i, row in enumerate(self.rows) if not self.is_reviewed(row)), None)
        self.image_index = self.image_names.index(self.rows[first_pending]["image"]) if first_pending is not None else 0
        self.root.title("痤疮病灶框严重程度标注（0*–IV）")
        self.root.geometry("1480x960")
        self.root.minsize(1080, 700)
        self._build_ui()
        self._bind_keys()
        self.root.after(120, lambda: self.show_image(self.image_index, first_pending))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="← 上一张", command=lambda: self.change_image(-1)).pack(side="left")
        ttk.Button(toolbar, text="下一张 →", command=lambda: self.change_image(1)).pack(side="left", padx=(5, 12))
        ttk.Button(toolbar, text="缩小 −", command=lambda: self.change_zoom(1 / 1.25)).pack(side="left")
        ttk.Button(toolbar, text="放大 +", command=lambda: self.change_zoom(1.25)).pack(side="left", padx=5)
        self.ids_button = ttk.Button(toolbar, text="隐藏编号", command=self.toggle_ids)
        self.ids_button.pack(side="left", padx=(0, 12))
        self.add_box_button = ttk.Button(toolbar, text="手动补框", command=self.toggle_add_box_mode)
        self.add_box_button.pack(side="left", padx=(0, 5))
        self.redraw_box_button = ttk.Button(toolbar, text="重画当前框", command=self.toggle_redraw_box_mode)
        self.redraw_box_button.pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="撤销删框", command=self.undo_delete).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="保存", command=self.save).pack(side="left")
        self.status_var = tk.StringVar(value="正在载入…")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right", padx=6)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        canvas_frame = ttk.Frame(body)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#15191f", highlightthickness=0, width=900, height=760)
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        side = ttk.Frame(body, width=365)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        ttk.Label(side, text="逐框选择项目类别", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            side,
            text="左侧看当前照片，右侧小图显示同一区域另一时点，辅助检查漏检。可补框、删误检、改框位置并修正等级。",
            wraplength=350,
        ).pack(anchor="w", pady=(3, 5))
        self.source_info_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.source_info_var, wraplength=350, foreground="#6B7280").pack(anchor="w", pady=(0, 5))
        ttk.Label(side, text="同一区域另一时点（框参照）", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        self.companion_label = ttk.Label(side, anchor="center", relief="groove", text="未配置配对照片")
        self.companion_label.pack(fill="x", ipady=2, pady=(2, 5))

        list_frame = ttk.Frame(side)
        list_frame.pack(fill="x")
        self.box_list = tk.Listbox(list_frame, height=6, exportselection=False, font=("Microsoft YaHei UI", 10))
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.box_list.yview)
        self.box_list.configure(yscrollcommand=list_scroll.set)
        self.box_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.box_list.bind("<<ListboxSelect>>", self.on_list_select)
        self.box_list.bind("<Up>", lambda _event: self.step_box(-1) or "break")
        self.box_list.bind("<Down>", lambda _event: self.step_box(1) or "break")

        self.crop_title_var = tk.StringVar(value="当前框局部放大")
        ttk.Label(side, textvariable=self.crop_title_var).pack(anchor="w", pady=(6, 2))
        self.model_suggestion_var = tk.StringVar(value="模型建议：未加载")
        ttk.Label(
            side,
            textvariable=self.model_suggestion_var,
            wraplength=350,
            foreground="#245A83",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 3))
        self.accept_suggestion_button = ttk.Button(
            side,
            text="采用当前框的模型建议",
            command=self.accept_model_suggestion,
            state="disabled",
        )
        self.accept_suggestion_button.pack(fill="x", pady=(0, 4))
        self.crop_label = ttk.Label(side, anchor="center", relief="groove")
        self.crop_label.pack(fill="x", ipady=3)

        ttk.Label(side, text="框内皮损等级（按数字键 0–4）", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(7, 3))
        for key, (grade, label) in zip(GRADE_KEYS, GRADES):
            ttk.Button(side, text=f"{key}　{label}", command=lambda value=grade: self.assign_grade(value)).pack(fill="x", pady=1)

        actions = ttk.Frame(side)
        actions.pack(fill="x", pady=(5, 2))
        ttk.Button(actions, text="U　不确定", command=self.mark_uncertain).pack(side="left", fill="x", expand=True)
        ttk.Button(actions, text="清除等级", command=self.clear_grade).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ttk.Button(side, text="删除当前框（误检）", command=self.delete_current_box).pack(fill="x", pady=(2, 2))
        ttk.Button(side, text="跳到下一个不确定框", command=self.go_next_uncertain).pack(fill="x", pady=(2, 4))
        ttk.Label(
            side,
            text="逐框操作映射：0* 仅痘印/瘢痕（排除，不计活动病灶）；I 粉刺；II 丘疹；III 脓疱；IV 结节/囊肿。一个框含多种皮损时按最高等级。",
            wraplength=350,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            side,
            text="快捷键：0–4 选级　U 不确定　Backspace 清除\n↑↓ 切框　←→ 切图片　N 补框　Ctrl+Z 撤销删框\n位置错：选择框后点“重画当前框”并拖出新框。",
            wraplength=350,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            side,
            text="圈线颜色：0* 灰、I 青、II 橙、III 红、IV 紫。未确认框显示模型建议色，黄色圆框表示当前选中。",
            wraplength=350,
            justify="left",
            foreground="#536273",
        ).pack(anchor="w", pady=(5, 0))

    def _bind_keys(self) -> None:
        for key, grade in GRADE_KEYS.items():
            self.root.bind(key, lambda _event, value=grade: self.assign_grade(value))
        self.root.bind("u", lambda _event: self.mark_uncertain())
        self.root.bind("U", lambda _event: self.mark_uncertain())
        self.root.bind("<BackSpace>", lambda _event: self.clear_grade())
        self.root.bind("<Up>", lambda _event: self.step_box(-1))
        self.root.bind("<Down>", lambda _event: self.step_box(1))
        self.root.bind("<Left>", lambda _event: self.change_image(-1))
        self.root.bind("<Right>", lambda _event: self.change_image(1))
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Control-z>", lambda _event: self.undo_delete())
        self.root.bind("+", lambda _event: self.change_zoom(1.25))
        self.root.bind("=", lambda _event: self.change_zoom(1.25))
        self.root.bind("-", lambda _event: self.change_zoom(1 / 1.25))
        self.root.bind("i", lambda _event: self.toggle_ids())
        self.root.bind("I", lambda _event: self.toggle_ids())
        self.root.bind("n", lambda _event: self.toggle_add_box_mode())
        self.root.bind("N", lambda _event: self.toggle_add_box_mode())
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_wheel)

    @staticmethod
    def is_reviewed(row: dict[str, str]) -> bool:
        return bool((row.get("box_severity_grade") or "").strip() or (row.get("review_note") or "").strip())

    def model_suggestion(self, row: dict[str, str]) -> dict[str, str] | None:
        return self.suggestions_by_key.get((row.get("image", ""), str(row.get("box_id", ""))))

    def show_image(self, image_index: int, preferred_row: int | None = None) -> None:
        self.image_index = max(0, min(image_index, len(self.image_names) - 1))
        self.current_image_name = self.image_names[self.image_index]
        image_path = self.image_dir / self.current_image_name
        if not image_path.is_file():
            messagebox.showerror("找不到图片", f"图片不存在：\n{image_path}", parent=self.root)
            return
        with Image.open(image_path) as opened:
            self.original_image = opened.convert("RGB")
        self.current_rows = self.rows_by_image[self.current_image_name]
        source_grade = next((row.get("source_filename_grade", "") for row in (self.rows[i] for i in self.current_rows)), "")
        source_filename = self.source_filename_by_image.get(self.current_image_name, "")
        if source_filename:
            self.source_info_var.set(f"{self.current_image_name}　·　来源图片 {source_filename}")
        else:
            self.source_info_var.set(f"{self.current_image_name}　·　原文件名等级 {source_grade or '未知'}（仅供记录）")
        self.update_companion_preview()
        self._populate_box_list(preferred_row)
        self.render_canvas(reset_view=True)

    def update_companion_preview(self) -> None:
        companion_name = self.companion_by_image.get(self.current_image_name, "")
        companion_path = self.image_dir / companion_name if companion_name else None
        if companion_path is None or not companion_path.is_file():
            self.companion_photo = None
            self.companion_label.configure(image="", text="未配置配对照片")
            return
        with Image.open(companion_path) as opened:
            preview = opened.convert("RGB")
        draw = ImageDraw.Draw(preview)
        width, height = preview.size
        for global_index in self.rows_by_image.get(companion_name, []):
            row = self.rows[global_index]
            x1, y1, x2, y2 = box_coordinates(row, width, height)
            color = GRADE_COLORS.get((row.get("box_severity_grade") or "").strip(), "#F2B705")
            draw.ellipse((x1, y1, x2, y2), outline=color, width=max(2, min(width, height) // 180))
            draw.text((max(0, x1), max(0, y1 - 14)), str(row.get("box_id", "")), fill=color)
        preview.thumbnail((300, 150), Image.Resampling.LANCZOS)
        self.companion_photo = ImageTk.PhotoImage(preview)
        self.companion_label.configure(image=self.companion_photo, text="")

    def _populate_box_list(self, preferred_row: int | None = None) -> None:
        self.box_list.delete(0, tk.END)
        for global_index in self.current_rows:
            row = self.rows[global_index]
            grade = (row.get("box_severity_grade") or "").strip()
            note = (row.get("review_note") or "").strip()
            suggestion = self.model_suggestion(row)
            if grade:
                label = f"已核 {grade}"
            elif note:
                label = "不确定"
            elif suggestion:
                model_grade = (suggestion.get("suggested_grade") or "").strip()
                model_conf = float(suggestion.get("top1_confidence") or 0)
                label = f"待核 建议{model_grade} {model_conf:.2f}"
            else:
                label = "待标"
            confidence = (row.get("confidence") or "").strip()
            confidence_text = f"　det {float(confidence):.2f}" if confidence else ""
            detection_source = (row.get("detection_source") or "").strip()
            if detection_source == "zoom_refined":
                source_text = "　局部复检"
            elif detection_source == "large_region_split":
                source_text = "　大框已细分"
            elif detection_source == "large_box_needs_review":
                source_text = "　大框需拆分"
            else:
                source_text = ""
            self.box_list.insert(tk.END, f"#{int(row['box_id']):02d}　{label}{confidence_text}{source_text}")
        if not self.current_rows:
            self.current_row_index = None
            self.update_crop()
            return
        if preferred_row in self.current_rows:
            selected = preferred_row
        else:
            selected = next((i for i in self.current_rows if not self.is_reviewed(self.rows[i])), self.current_rows[0])
        self.select_row(selected, render=False)

    def select_row(self, global_index: int, render: bool = True) -> None:
        if global_index not in self.current_rows:
            return
        self.current_row_index = global_index
        local_index = self.current_rows.index(global_index)
        self.box_list.selection_clear(0, tk.END)
        self.box_list.selection_set(local_index)
        self.box_list.see(local_index)
        if render:
            self.render_canvas()
        self.update_crop()
        self.update_status()

    def on_list_select(self, _event: tk.Event) -> None:
        selection = self.box_list.curselection()
        if selection:
            self.select_row(self.current_rows[selection[0]])

    def render_canvas(self, reset_view: bool = False) -> None:
        if self.original_image is None:
            return
        old_xview = self.canvas.xview()[0] if self.canvas.xview() else 0.0
        old_yview = self.canvas.yview()[0] if self.canvas.yview() else 0.0
        self.canvas.update_idletasks()
        viewport_width = max(400, self.canvas.winfo_width() - 8)
        viewport_height = max(400, self.canvas.winfo_height() - 8)
        image_width, image_height = self.original_image.size
        fit_scale = min(viewport_width / image_width, viewport_height / image_height)
        self.display_scale = max(0.05, fit_scale * self.zoom)
        display_size = (max(1, round(image_width * self.display_scale)), max(1, round(image_height * self.display_scale)))
        rendered = self.original_image.resize(display_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rendered)
        font_size = max(10, round(24 * self.display_scale))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        line_width = max(2, round(3 * self.display_scale))
        for global_index in self.current_rows:
            row = self.rows[global_index]
            x1, y1, x2, y2 = box_coordinates(row, image_width, image_height)
            coords = tuple(round(value * self.display_scale) for value in (x1, y1, x2, y2))
            grade = (row.get("box_severity_grade") or "").strip()
            note = (row.get("review_note") or "").strip()
            suggestion = self.model_suggestion(row)
            if grade:
                color_key = grade
            elif note:
                color_key = "uncertain"
            elif suggestion:
                color_key = (suggestion.get("suggested_grade") or "").strip()
            else:
                color_key = "unlabeled"
            color = GRADE_COLORS.get(color_key, "#F2B705")
            selected = global_index == self.current_row_index
            draw.ellipse(coords, outline="#FFF000" if selected else color, width=line_width + (2 if selected else 0))
            if self.show_ids:
                tag = str(row["box_id"])
                tag_box = draw.textbbox((0, 0), tag, font=font)
                tag_width, tag_height = tag_box[2] - tag_box[0], tag_box[3] - tag_box[1]
                pad = max(2, round(4 * self.display_scale))
                tx, ty = max(0, coords[0]), max(0, coords[1] - tag_height - 2 * pad)
                draw.rectangle((tx, ty, tx + tag_width + 2 * pad, ty + tag_height + 2 * pad), fill="#102A43")
                draw.text((tx + pad, ty + pad - tag_box[1]), tag, fill="white", font=font)
        self.canvas_photo = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, *display_size))
        if reset_view:
            self.canvas.xview_moveto(0.0)
            self.canvas.yview_moveto(0.0)
        else:
            self.canvas.xview_moveto(old_xview)
            self.canvas.yview_moveto(old_yview)
        self.update_status()

    def update_crop(self) -> None:
        if self.original_image is None or self.current_row_index is None:
            self.crop_photo = None
            self.crop_label.configure(image="", text="没有选中的框")
            self.crop_title_var.set("当前框局部放大")
            self.model_suggestion_var.set("模型建议：未选择框")
            self.accept_suggestion_button.configure(state="disabled")
            return
        row = self.rows[self.current_row_index]
        image_width, image_height = self.original_image.size
        x1, y1, x2, y2 = box_coordinates(row, image_width, image_height)
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        margin_x, margin_y = max(24, box_width * 1.4), max(24, box_height * 1.4)
        left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
        right, bottom = min(image_width, int(x2 + margin_x)), min(image_height, int(y2 + margin_y))
        crop = self.original_image.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.ellipse((x1 - left, y1 - top, x2 - left, y2 - top), outline="#FFF000", width=max(2, min(crop.size) // 100))
        scale = min(320 / max(1, crop.width), 205 / max(1, crop.height))
        preview = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        self.crop_photo = ImageTk.PhotoImage(preview)
        self.crop_label.configure(image=self.crop_photo, text="")
        row_grade = (row.get("box_severity_grade") or "").strip()
        label = GRADE_NAMES.get(row_grade, row_grade) if row_grade else ("不确定" if row.get("review_note") else "待标")
        confidence = (row.get("confidence") or "").strip()
        confidence_text = f"　conf {float(confidence):.2f}" if confidence else ""
        detection_source = (row.get("detection_source") or "").strip()
        source_labels = {
            "zoom_refined": "局部放大复检",
            "large_region_split": "大框拆成小框",
            "large_box_needs_review": "大框需拆分复核",
        }
        source_text = f"　·　{source_labels[detection_source]}" if detection_source in source_labels else ""
        self.crop_title_var.set(f"框 #{row['box_id']}　{label}{confidence_text}{source_text}　·　局部放大")
        suggestion = self.model_suggestion(row)
        if suggestion:
            suggested_grade = (suggestion.get("suggested_grade") or "").strip()
            suggested_confidence = float(suggestion.get("top1_confidence") or 0)
            review_state = f"人工已选 {row_grade}" if row_grade else ("已标不确定" if row.get("review_note") else "待确认")
            self.model_suggestion_var.set(
                f"模型建议：{suggested_grade}（分类分数 {suggested_confidence:.2f}）　·　{review_state}"
            )
            button_state = "normal" if not row_grade and not row.get("review_note") else "disabled"
            self.accept_suggestion_button.configure(state=button_state)
        else:
            self.model_suggestion_var.set("模型建议：无")
            self.accept_suggestion_button.configure(state="disabled")

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        if self.add_box_mode or self.redraw_box_mode:
            self.drag_start = self._event_image_coordinates(event)
            canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.temp_rectangle = self.canvas.create_oval(
                canvas_x, canvas_y, canvas_x, canvas_y, outline="#FFF000", width=3, dash=(5, 3)
            )
            return
        self.select_box_at(event)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if (self.add_box_mode or self.redraw_box_mode) and self.drag_start is not None and self.temp_rectangle is not None:
            start_x = self.drag_start[0] * self.display_scale
            start_y = self.drag_start[1] * self.display_scale
            end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.canvas.coords(self.temp_rectangle, start_x, start_y, end_x, end_y)

    def on_canvas_release(self, event: tk.Event) -> None:
        if not (self.add_box_mode or self.redraw_box_mode) or self.drag_start is None:
            return
        start_x, start_y = self.drag_start
        end_x, end_y = self._event_image_coordinates(event)
        if self.temp_rectangle is not None:
            self.canvas.delete(self.temp_rectangle)
        self.temp_rectangle = None
        self.drag_start = None
        if self.original_image is None:
            return
        image_width, image_height = self.original_image.size
        start_x, end_x = (max(0.0, min(value, image_width)) for value in (start_x, end_x))
        start_y, end_y = (max(0.0, min(value, image_height)) for value in (start_y, end_y))
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            self.render_canvas()
            return
        if self.redraw_box_mode:
            self.redraw_current_box(x1, y1, x2, y2)
        else:
            self.add_manual_box(x1, y1, x2, y2)

    def _event_image_coordinates(self, event: tk.Event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x) / self.display_scale
        y = self.canvas.canvasy(event.y) / self.display_scale
        if self.original_image is None:
            return x, y
        width, height = self.original_image.size
        return max(0.0, min(x, width)), max(0.0, min(y, height))

    def select_box_at(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        x, y = self._event_image_coordinates(event)
        image_width, image_height = self.original_image.size
        contained: list[tuple[float, int]] = []
        nearest: tuple[float, int] | None = None
        for global_index in self.current_rows:
            x1, y1, x2, y2 = box_coordinates(self.rows[global_index], image_width, image_height)
            if x1 <= x <= x2 and y1 <= y <= y2:
                contained.append(((x2 - x1) * (y2 - y1), global_index))
            distance = (((x1 + x2) / 2 - x) ** 2 + ((y1 + y2) / 2 - y) ** 2) ** 0.5
            if nearest is None or distance < nearest[0]:
                nearest = (distance, global_index)
        if contained:
            self.select_row(min(contained)[1])
        elif nearest and nearest[0] * self.display_scale <= 32:
            self.select_row(nearest[1])

    def add_manual_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.original_image is None:
            return
        image_width, image_height = self.original_image.size
        box_id = self.next_box_id_by_image.get(self.current_image_name, 1)
        self.next_box_id_by_image[self.current_image_name] = box_id + 1
        source_grade = next(
            (self.rows[index].get("source_filename_grade", "") for index in self.current_rows), ""
        )
        new_row = {name: "" for name in self.headers}
        new_row.update(
            {
                "image": self.current_image_name,
                "source_filename_grade": source_grade,
                "box_id": str(box_id),
                "source_class_id": "0",
                "x_center_norm": str(((x1 + x2) / 2) / image_width),
                "y_center_norm": str(((y1 + y2) / 2) / image_height),
                "width_norm": str((x2 - x1) / image_width),
                "height_norm": str((y2 - y1) / image_height),
                "confidence": "",
                "box_severity_grade": "",
                "review_note": "",
            }
        )
        self.rows.append(new_row)
        new_index = len(self.rows) - 1
        self.rows_by_image.setdefault(self.current_image_name, []).append(new_index)
        self.current_rows = self.rows_by_image[self.current_image_name]
        self.save(show_message=False)
        self._populate_box_list(preferred_row=new_index)
        self.render_canvas()

    def redraw_current_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.original_image is None or self.current_row_index is None:
            return
        image_width, image_height = self.original_image.size
        row = self.rows[self.current_row_index]
        row.update({
            "x_center_norm": str(((x1 + x2) / 2) / image_width),
            "y_center_norm": str(((y1 + y2) / 2) / image_height),
            "width_norm": str((x2 - x1) / image_width),
            "height_norm": str((y2 - y1) / image_height),
        })
        if "detection_source" in row:
            row["detection_source"] = "manual_adjusted"
        self.redraw_box_mode = False
        self.redraw_box_button.configure(text="重画当前框")
        self.save(show_message=False)
        self._populate_box_list(preferred_row=self.current_row_index)
        self.render_canvas()

    def accept_model_suggestion(self) -> None:
        if self.current_row_index is None:
            return
        suggestion = self.model_suggestion(self.rows[self.current_row_index])
        if not suggestion:
            return
        grade = (suggestion.get("suggested_grade") or "").strip()
        if grade in GRADE_NAMES:
            self.assign_grade(grade)

    def assign_grade(self, grade: str) -> None:
        if self.current_row_index is None:
            return
        row = self.rows[self.current_row_index]
        row["box_severity_grade"] = grade
        if (row.get("review_note") or "").strip().lower() == "uncertain":
            row["review_note"] = ""
        self.save(show_message=False)
        self._populate_box_list(preferred_row=self.current_row_index)
        self.advance_to_unreviewed()

    def mark_uncertain(self) -> None:
        if self.current_row_index is None:
            return
        row = self.rows[self.current_row_index]
        row["box_severity_grade"] = ""
        row["review_note"] = "uncertain"
        self.save(show_message=False)
        self._populate_box_list(preferred_row=self.current_row_index)
        self.advance_to_unreviewed()

    def clear_grade(self) -> None:
        if self.current_row_index is None:
            return
        self.rows[self.current_row_index]["box_severity_grade"] = ""
        self.rows[self.current_row_index]["review_note"] = ""
        self.save(show_message=False)
        self._populate_box_list(preferred_row=self.current_row_index)

    def delete_current_box(self) -> None:
        if self.current_row_index is None:
            return
        row = self.rows[self.current_row_index]
        box_id = row["box_id"]
        if not messagebox.askyesno(
            "删除误检框",
            f"删除 {self.current_image_name} 的框 #{box_id}？\n可用“撤销删框”恢复。",
            parent=self.root,
        ):
            return
        old_index = self.current_row_index
        old_local = self.current_rows.index(old_index)
        removed = self.rows.pop(old_index)
        self.deleted_rows.append((old_index, removed))
        self._rebuild_rows_by_image()
        self.current_rows = self.rows_by_image[self.current_image_name]
        preferred = self.current_rows[min(old_local, len(self.current_rows) - 1)] if self.current_rows else None
        self.current_row_index = None
        self.save(show_message=False)
        self._populate_box_list(preferred_row=preferred)
        self.render_canvas()

    def undo_delete(self) -> None:
        if not self.deleted_rows:
            return
        row_index, row = self.deleted_rows.pop()
        self.rows.insert(min(row_index, len(self.rows)), row)
        self._rebuild_rows_by_image()
        image_name = row["image"]
        image_index = self.image_names.index(image_name)
        self.show_image(image_index, row_index)
        self.save(show_message=False)

    def _rebuild_rows_by_image(self) -> None:
        self.rows_by_image = {name: [] for name in self.image_names}
        for index, row in enumerate(self.rows):
            self.rows_by_image.setdefault(row["image"], []).append(index)

    def toggle_add_box_mode(self) -> None:
        self.add_box_mode = not self.add_box_mode
        self.redraw_box_mode = False
        self.redraw_box_button.configure(text="重画当前框")
        self.add_box_button.configure(text="取消补框" if self.add_box_mode else "手动补框")
        if not self.add_box_mode:
            self.drag_start = None
            if self.temp_rectangle is not None:
                self.canvas.delete(self.temp_rectangle)
                self.temp_rectangle = None
        self.update_status()

    def toggle_redraw_box_mode(self) -> None:
        if self.current_row_index is None:
            messagebox.showinfo("先选一个框", "点击左侧照片上的框或右侧列表中的框，然后拖动新位置重画。", parent=self.root)
            return
        self.redraw_box_mode = not self.redraw_box_mode
        self.add_box_mode = False
        self.add_box_button.configure(text="手动补框")
        self.redraw_box_button.configure(text="取消重画" if self.redraw_box_mode else "重画当前框")
        if not self.redraw_box_mode:
            self.drag_start = None
            if self.temp_rectangle is not None:
                self.canvas.delete(self.temp_rectangle)
                self.temp_rectangle = None
        self.update_status()

    def advance_to_unreviewed(self) -> None:
        if self.current_row_index is None:
            return
        current = self.current_row_index
        for offset in range(1, len(self.rows) + 1):
            candidate = (current + offset) % len(self.rows)
            if not self.is_reviewed(self.rows[candidate]):
                target_image_index = self.image_names.index(self.rows[candidate]["image"])
                if target_image_index == self.image_index:
                    self.select_row(candidate)
                else:
                    self.show_image(target_image_index, candidate)
                return
        self.update_status()

    def go_next_uncertain(self) -> None:
        if self.current_row_index is None:
            return
        for offset in range(1, len(self.rows) + 1):
            candidate = (self.current_row_index + offset) % len(self.rows)
            if (self.rows[candidate].get("review_note") or "").strip().lower() == "uncertain":
                target_image_index = self.image_names.index(self.rows[candidate]["image"])
                if target_image_index == self.image_index:
                    self.select_row(candidate)
                else:
                    self.show_image(target_image_index, candidate)
                return
        messagebox.showinfo("没有待复核框", "当前没有标记为“不确定”的框。", parent=self.root)

    def step_box(self, offset: int) -> None:
        if not self.current_rows:
            return
        local = self.current_rows.index(self.current_row_index) if self.current_row_index in self.current_rows else 0
        next_local = max(0, min(local + offset, len(self.current_rows) - 1))
        self.select_row(self.current_rows[next_local])

    def change_image(self, offset: int) -> None:
        self.show_image(self.image_index + offset)

    def change_zoom(self, multiplier: float) -> None:
        self.zoom = max(0.5, min(4.0, self.zoom * multiplier))
        self.render_canvas()

    def on_ctrl_wheel(self, event: tk.Event) -> str:
        self.change_zoom(1.2 if event.delta > 0 else 1 / 1.2)
        return "break"

    def toggle_ids(self) -> None:
        self.show_ids = not self.show_ids
        self.ids_button.configure(text="隐藏编号" if self.show_ids else "显示编号")
        self.render_canvas()

    def save(self, show_message: bool = True) -> None:
        temp_path = self.csv_path.with_name(self.csv_path.name + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temp_path, self.csv_path)
        self.update_status()
        if show_message:
            messagebox.showinfo("已保存", f"逐框等级已保存到：\n{self.csv_path}", parent=self.root)

    def update_status(self) -> None:
        completed = sum(self.is_reviewed(row) for row in self.rows)
        uncertain = sum((row.get("review_note") or "").strip().lower() == "uncertain" for row in self.rows)
        if self.current_row_index is None or not self.current_rows:
            current = "无框"
        else:
            local = self.current_rows.index(self.current_row_index) + 1
            row = self.rows[self.current_row_index]
            grade = (row.get("box_severity_grade") or "").strip()
            label = GRADE_NAMES.get(grade, grade) if grade else ("不确定" if row.get("review_note") else "待标")
            current = f"框 {local}/{len(self.current_rows)} · {label}"
        if not self.rows:
            progress = "待从图片上手动补框"
        elif completed == len(self.rows) and uncertain == 0:
            progress = f"全部定级 {completed}/{len(self.rows)} 框"
        else:
            progress = f"已处理 {completed}/{len(self.rows)} 框 · 待复核 {uncertain}"
        mode = "　补框模式：拖动绘框" if self.add_box_mode else ("　重画模式：拖动框出新位置" if self.redraw_box_mode else "")
        if self.current_image_name:
            self.status_var.set(f"图片 {self.image_index + 1}/{len(self.image_names)}　{current}　{progress}{mode}")
        else:
            self.status_var.set(progress)


def main() -> None:
    parser = argparse.ArgumentParser(description="逐个 YOLO 框标注项目定义的痤疮病灶类别 0*–IV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="标注表路径")
    parser.add_argument("--suggestions", type=Path, default=None, help="可选的逐框模型建议 CSV")
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else (ROOT / args.csv).resolve()
    suggestions_path = args.suggestions
    if suggestions_path is not None and not suggestions_path.is_absolute():
        suggestions_path = (ROOT / suggestions_path).resolve()
    if not csv_path.is_file():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("找不到标注表", f"找不到文件：\n{csv_path}", parent=root)
        root.destroy()
        return
    root = tk.Tk()
    try:
        BoxSeverityAnnotator(root, csv_path, suggestions_path)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("标注工具无法启动", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
