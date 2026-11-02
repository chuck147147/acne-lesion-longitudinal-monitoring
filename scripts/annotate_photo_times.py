"""Review project photos and record known capture date/order without guessing."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "subject_split_manifest.csv"
TIME_FIELDS = ["session_id", "capture_date", "capture_time", "time_precision", "sequence_order", "notes"]
PRECISION_LABELS = {
    "": "未填写",
    "exact": "准确到日/时",
    "approximate": "大致时间",
    "relative_order": "只能排先后",
    "unknown": "时间未知",
}
PRECISION_CODES = {label: code for code, label in PRECISION_LABELS.items()}
EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


def read_exif_datetime(path: Path) -> tuple[str, str] | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except OSError:
        return None
    for tag in EXIF_DATE_TAGS:
        value = str(exif.get(tag) or "").strip()
        match = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2})$", value)
        if match:
            year, month, day, time_text = match.groups()
            return f"{year}-{month}-{day}", time_text
    return None


class PhotoTimeAnnotator:
    def __init__(self, root: tk.Tk, manifest_path: Path) -> None:
        self.root = root
        self.manifest_path = manifest_path
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.headers = list(reader.fieldnames or [])
            self.rows = [dict(row) for row in reader]
        required = {"image_id", "relative_path", "participant_id", "split"}
        missing = required - set(self.headers)
        if missing:
            raise ValueError(f"受试者清单缺少列：{', '.join(sorted(missing))}")
        for field in TIME_FIELDS:
            if field not in self.headers:
                self.headers.append(field)
            for row in self.rows:
                row.setdefault(field, "")
        if not self.rows:
            raise ValueError("受试者清单中没有照片")

        self.index = 0
        self.current_photo: ImageTk.PhotoImage | None = None
        self.status_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.exif_var = tk.StringVar(value="未发现 EXIF 拍摄时间")
        self.session_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.precision_var = tk.StringVar()
        self.order_var = tk.StringVar()
        self.notes_var = tk.StringVar()

        self.root.title("照片拍摄时间标注")
        self.root.geometry("1260x820")
        self.root.minsize(1000, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        self.show_current()

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.title_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.title_var, font=("Microsoft YaHei UI", 15, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.info_var, foreground="#5B6472").pack(side="right")

        body = ttk.Frame(main)
        body.grid(row=1, column=0, columnspan=2, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        image_frame = ttk.LabelFrame(body, text="照片", padding=6)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        form = ttk.LabelFrame(body, text="时间信息", padding=12)
        form.grid(row=0, column=1, sticky="ns")
        form.columnconfigure(0, weight=1)
        ttk.Label(form, text="拍摄会话（同一时段用同一编号）").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.session_var, width=34).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        ttk.Label(form, text="拍摄日期 / 大致时间").grid(row=2, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.date_var, width=34).grid(row=3, column=0, sticky="ew", pady=(3, 3))
        ttk.Label(form, text="可填 YYYY-MM-DD、YYYY-MM，或‘某月上旬’").grid(row=4, column=0, sticky="w", pady=(0, 10))

        ttk.Label(form, text="拍摄时刻（记得时填写）").grid(row=5, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.time_var, width=34).grid(row=6, column=0, sticky="ew", pady=(3, 12))

        ttk.Label(form, text="时间精度").grid(row=7, column=0, sticky="w")
        self.precision_combo = ttk.Combobox(
            form,
            textvariable=self.precision_var,
            values=list(PRECISION_CODES),
            state="readonly",
            width=31,
        )
        self.precision_combo.grid(row=8, column=0, sticky="ew", pady=(3, 12))

        ttk.Label(form, text="若只知道先后，填顺序号（越小越早）").grid(row=9, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.order_var, width=34).grid(row=10, column=0, sticky="ew", pady=(3, 12))

        ttk.Label(form, textvariable=self.exif_var, wraplength=320, foreground="#5B6472").grid(
            row=11, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Button(form, text="采用 EXIF 候选时间", command=self.use_exif).grid(row=12, column=0, sticky="ew", pady=(0, 12))

        ttk.Label(form, text="备注 / 时间来源").grid(row=13, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.notes_var, width=34).grid(row=14, column=0, sticky="ew", pady=(3, 10))

        ttk.Label(
            form,
            text="只填写你确认或记得的时间；不知道可留空或选‘时间未知’。不要按痘痘轻重推测先后。",
            wraplength=320,
            justify="left",
        ).grid(row=15, column=0, sticky="w", pady=(0, 14))

        nav = ttk.Frame(form)
        nav.grid(row=16, column=0, sticky="ew")
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="上一张", command=self.previous).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(nav, text="保存", command=self.save_current).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(form, text="保存并下一张", command=self.next).grid(row=17, column=0, sticky="ew", pady=(8, 0))

        footer = ttk.Frame(main)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var, foreground="#247A55").pack(side="left")
        ttk.Label(footer, text=f"保存到：{self.manifest_path}", foreground="#687386").pack(side="right")

    def image_path(self, row: dict[str, str]) -> Path:
        return ROOT / (row.get("relative_path") or "")

    def show_current(self) -> None:
        row = self.rows[self.index]
        image_path = self.image_path(row)
        self.title_var.set(f"照片 {self.index + 1}/{len(self.rows)}　{row.get('image_id', '')}")
        self.info_var.set(
            f"受试者：{row.get('participant_id') or '未填'}　·　集合：{row.get('split') or '未填'}　·　{image_path.name}"
        )
        self.session_var.set(row.get("session_id", ""))
        self.date_var.set(row.get("capture_date", ""))
        self.time_var.set(row.get("capture_time", ""))
        precision = row.get("time_precision", "")
        self.precision_var.set(PRECISION_LABELS.get(precision, "未填写"))
        self.order_var.set(row.get("sequence_order", ""))
        self.notes_var.set(row.get("notes", ""))

        exif_time = read_exif_datetime(image_path) if image_path.is_file() else None
        self._exif_candidate = exif_time
        if exif_time:
            self.exif_var.set(f"EXIF 候选：{exif_time[0]} {exif_time[1]}（请核对后再采用）")
        else:
            self.exif_var.set("未发现 EXIF 拍摄时间")

        if not image_path.is_file():
            self.image_label.configure(image="", text=f"找不到照片：\n{image_path}")
            self.current_photo = None
            return
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            image.thumbnail((800, 720), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_photo, text="")
        except Exception as error:
            self.image_label.configure(image="", text=f"无法打开照片：{error}")
            self.current_photo = None

    def use_exif(self) -> None:
        if not self._exif_candidate:
            messagebox.showinfo("没有 EXIF 时间", "这张照片没有可用的 EXIF 拍摄时间。", parent=self.root)
            return
        date_text, time_text = self._exif_candidate
        self.date_var.set(date_text)
        self.time_var.set(time_text)
        self.precision_var.set(PRECISION_LABELS["exact"])
        if not self.notes_var.get().strip():
            self.notes_var.set("采用 EXIF 候选时间，已人工核对")

    def write_manifest(self) -> None:
        temp_path = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temp_path, self.manifest_path)

    def save_current(self, show_status: bool = True) -> bool:
        row = self.rows[self.index]
        row["session_id"] = self.session_var.get().strip()
        row["capture_date"] = self.date_var.get().strip()
        row["capture_time"] = self.time_var.get().strip()
        row["time_precision"] = PRECISION_CODES.get(self.precision_var.get(), "")
        row["sequence_order"] = self.order_var.get().strip()
        row["notes"] = self.notes_var.get().strip()
        try:
            self.write_manifest()
        except OSError as error:
            messagebox.showerror("保存失败", str(error), parent=self.root)
            return False
        if show_status:
            self.status_var.set(f"已保存 {row.get('image_id', '')} 的时间信息")
        return True

    def previous(self) -> None:
        self.save_current(show_status=False)
        self.index = (self.index - 1) % len(self.rows)
        self.show_current()

    def next(self) -> None:
        if not self.save_current(show_status=False):
            return
        self.index = (self.index + 1) % len(self.rows)
        self.show_current()

    def close(self) -> None:
        if self.save_current(show_status=False):
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="逐张照片填写已知拍摄日期、会话和相对顺序")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到受试者清单：{manifest_path}")
    root = tk.Tk()
    try:
        PhotoTimeAnnotator(root, manifest_path)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("时间标注工具无法启动", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
