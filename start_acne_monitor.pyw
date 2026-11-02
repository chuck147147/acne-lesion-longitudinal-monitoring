"""Windows desktop workbench for the three-view acne follow-up workflow."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "monitor_three_photos.py"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOCAL_CONFIG = ROOT / "monitor.local.json"
OUTPUTS_DIR = ROOT / "outputs" / "acne_monitor_runs"
REGIONS = ("左脸", "右脸", "下巴")
IMAGE_TYPES = [("图像", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]


def order_named_views(paths: list[str]) -> list[str] | None:
    if len(paths) != 3:
        return None
    found: dict[int, str] = {}
    for path in paths:
        name = Path(path).stem.lower().replace("-", "_").replace(" ", "_")
        if "左脸" in name or "left_face" in name or "left_cheek" in name:
            index = 0
        elif "右脸" in name or "right_face" in name or "right_cheek" in name:
            index = 1
        elif "下巴" in name or "chin" in name or "jaw" in name:
            index = 2
        else:
            return None
        if index in found:
            return None
        found[index] = path
    return [found[index] for index in range(3)] if set(found) == {0, 1, 2} else None


class AcneMonitorDesktop:
    COLORS = {
        "canvas": "#F3F5F7",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFB",
        "text": "#18232D",
        "muted": "#64717D",
        "line": "#DCE3E8",
        "accent": "#197565",
        "accent_hover": "#135D51",
        "accent_soft": "#E8F3F0",
        "warning": "#8B5D12",
        "warning_soft": "#FBF3E5",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("痤疮长期监测（自动追踪）")
        self.root.geometry("1040x720")
        self.root.minsize(900, 640)
        self.root.configure(bg=self.COLORS["canvas"])

        self.current: list[str | None] = [None, None, None]
        self.baseline_run: Path | None = self.default_baseline_run()
        self.last_output: Path | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.photo_labels: list[ttk.Label] = []
        self.status_var = tk.StringVar(value="准备就绪")
        self.baseline_var = tk.StringVar(value=self.describe_previous())
        self.selection_var = tk.StringVar(value="已选择 0 / 3 张照片")

        self._configure_styles()
        self._build_layout()
        self._refresh_photo_rows()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("App.TFrame", background=self.COLORS["canvas"])
        style.configure("Panel.TFrame", background=self.COLORS["surface"])
        style.configure("Soft.TFrame", background=self.COLORS["surface_alt"])
        style.configure("App.TLabel", background=self.COLORS["canvas"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=self.COLORS["canvas"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=self.COLORS["canvas"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 10))
        style.configure("PanelTitle.TLabel", background=self.COLORS["surface"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Body.TLabel", background=self.COLORS["surface"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", background=self.COLORS["surface"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("PhotoName.TLabel", background=self.COLORS["surface_alt"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(16, 9), foreground="white", background=self.COLORS["accent"], borderwidth=0, focusthickness=2)
        style.map("Accent.TButton", background=[("active", self.COLORS["accent_hover"]), ("disabled", "#A7BDB8")], foreground=[("disabled", "#F6F8F8")])
        style.configure("Quiet.TButton", font=("Microsoft YaHei UI", 9), padding=(10, 6), background=self.COLORS["surface"], foreground=self.COLORS["text"])
        style.map("Quiet.TButton", background=[("active", self.COLORS["surface_alt"])])
        style.configure("Small.TButton", font=("Microsoft YaHei UI", 9), padding=(8, 4), background=self.COLORS["surface"], foreground=self.COLORS["text"])
        style.map("Small.TButton", background=[("active", self.COLORS["surface_alt"])])
        style.configure("Horizontal.TProgressbar", troughcolor=self.COLORS["line"], background=self.COLORS["accent"], bordercolor=self.COLORS["line"], lightcolor=self.COLORS["accent"], darkcolor=self.COLORS["accent"])

    @staticmethod
    def is_reviewed_run(run_dir: Path) -> bool:
        result_path = run_dir / "result.json"
        if not result_path.is_file() or not (run_dir / "lesions.csv").is_file():
            return False
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return bool(data.get("box_review_applied")) and len(data.get("current_images", [])) == 3 and len(data.get("regions", [])) == 3
        except (OSError, ValueError, TypeError):
            return False

    @classmethod
    def default_baseline_run(cls) -> Path | None:
        if LOCAL_CONFIG.is_file():
            try:
                data = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
                configured = Path(data.get("baseline_run", ""))
                configured = configured if configured.is_absolute() else ROOT / configured
                if configured.is_dir() and cls.is_reviewed_run(configured):
                    return configured.resolve()
            except (OSError, ValueError, TypeError):
                pass
        if OUTPUTS_DIR.is_dir():
            for run_dir in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
                if run_dir.is_dir() and cls.is_reviewed_run(run_dir):
                    return run_dir.resolve()
        return None

    def describe_previous(self) -> str:
        if self.baseline_run is None:
            return "未选择基准。首次记录会建立基线，不计算前后变化。"
        try:
            data = json.loads((self.baseline_run / "result.json").read_text(encoding="utf-8"))
            names = [Path(path).name for path in data.get("current_images", [])]
        except (OSError, ValueError, TypeError):
            names = []
        session = self.baseline_run.name
        return f"最近已复核记录 · {session}\n" + "\n".join(f"{REGIONS[i]}：{name}" for i, name in enumerate(names))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 12))
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="痤疮长期监测", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="导入三个视角，复核本次病灶，再查看与上次记录的变化。照片和结果保存在本机。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=7, uniform="body")
        body.columnconfigure(1, weight=5, uniform="body")
        body.rowconfigure(0, weight=1)

        self._build_photo_panel(body)
        self._build_history_panel(body)
        self._build_footer(outer)

    def _panel(self, parent: ttk.Frame, column: int) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 10) if column == 0 else (10, 0))
        return panel

    def _build_photo_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 0)
        heading = ttk.Frame(panel, style="Panel.TFrame")
        heading.pack(fill="x")
        ttk.Label(heading, text="本次随访照片", style="PanelTitle.TLabel").pack(side="left")
        ttk.Label(heading, textvariable=self.selection_var, style="Muted.TLabel").pack(side="right", pady=(2, 0))
        ttk.Label(
            panel,
            text="每个视角选一张。文件名包含“左脸 / 右脸 / 下巴”时，批量选择可自动排序。",
            style="Muted.TLabel",
            wraplength=490,
        ).pack(anchor="w", pady=(7, 16))

        for index, region in enumerate(REGIONS):
            row = ttk.Frame(panel, style="Soft.TFrame", padding=(12, 11))
            row.pack(fill="x", pady=4)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=region, style="Body.TLabel", width=8).grid(row=0, column=0, sticky="w", padx=(0, 8))
            filename = ttk.Label(row, text="尚未选择", style="PhotoName.TLabel", wraplength=245)
            filename.grid(row=0, column=1, sticky="w", padx=(0, 6))
            self.photo_labels.append(filename)
            ttk.Button(row, text="选择照片", style="Small.TButton", command=lambda i=index: self.pick_region(i)).grid(row=0, column=2, sticky="e")

        controls = ttk.Frame(panel, style="Panel.TFrame")
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(controls, text="一次选择三张", style="Quiet.TButton", command=self.pick_current).pack(side="left")
        ttk.Button(controls, text="清空选择", style="Quiet.TButton", command=self.clear_current).pack(side="left", padx=(8, 0))

        ttk.Separator(panel).pack(fill="x", pady=(14, 11))
        ttk.Label(panel, text="复核流程", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            panel,
            text="模型会先检测并分类。随后只检查本次三张照片：可以补框、删误检、调整框位和修改类别。复核完成后自动生成对比总览和长期趋势。",
            style="Muted.TLabel",
            wraplength=490,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

    def _build_history_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 1)
        ttk.Label(panel, text="对比基准", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            panel,
            text="沿用上次人工复核的照片、框和等级，不会重复运行检测。",
            style="Muted.TLabel",
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(7, 12))

        baseline_box = tk.Frame(panel, bg=self.COLORS["surface_alt"], highlightbackground=self.COLORS["line"], highlightthickness=1, padx=12, pady=11)
        baseline_box.pack(fill="x")
        self.baseline_label = tk.Label(
            baseline_box,
            textvariable=self.baseline_var,
            bg=self.COLORS["surface_alt"],
            fg=self.COLORS["text"] if self.baseline_run else self.COLORS["muted"],
            justify="left",
            anchor="w",
            font=("Microsoft YaHei UI", 9),
            wraplength=315,
        )
        self.baseline_label.pack(fill="x")

        baseline_actions = ttk.Frame(panel, style="Panel.TFrame")
        baseline_actions.pack(fill="x", pady=(10, 0))
        ttk.Button(baseline_actions, text="更换基准", style="Quiet.TButton", command=self.pick_previous).pack(side="left")
        ttk.Button(baseline_actions, text="清除基准", style="Quiet.TButton", command=self.clear_previous).pack(side="left", padx=(8, 0))

        ttk.Separator(panel).pack(fill="x", pady=(14, 12))
        ttk.Label(panel, text="长期记录", style="PanelTitle.TLabel").pack(anchor="w")
        sessions = len([p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]) if OUTPUTS_DIR.is_dir() else 0
        self.sessions_label = ttk.Label(panel, text=f"本机已保存 {sessions} 次随访运行", style="Body.TLabel")
        self.sessions_label.pack(anchor="w", pady=(8, 4))
        ttk.Label(
            panel,
            text="每次完成复核后会汇总历史趋势。原始照片和输出文件留在本机项目目录中。",
            style="Muted.TLabel",
            wraplength=330,
            justify="left",
        ).pack(anchor="w")

        ttk.Separator(panel).pack(fill="x", pady=(14, 12))
        ttk.Label(panel, text="结果位置", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text=str(OUTPUTS_DIR), style="Muted.TLabel", wraplength=330, justify="left").pack(anchor="w", pady=(8, 10))
        ttk.Button(panel, text="打开结果文件夹", style="Quiet.TButton", command=self.open_output_folder).pack(anchor="w")

    def _build_footer(self, parent: ttk.Frame) -> None:
        footer = ttk.Frame(parent, style="App.TFrame")
        footer.pack(fill="x", pady=(10, 0))
        status_area = ttk.Frame(footer, style="App.TFrame")
        status_area.pack(side="left", fill="x", expand=True)
        self.status_label = ttk.Label(status_area, textvariable=self.status_var, style="Subtitle.TLabel", wraplength=520)
        self.status_label.pack(anchor="w")
        self.progress = ttk.Progressbar(status_area, mode="indeterminate", length=300)

        actions = ttk.Frame(footer, style="App.TFrame")
        actions.pack(side="right")
        self.history_button = ttk.Button(actions, text="查看长期趋势", style="Quiet.TButton", command=self.view_history)
        self.history_button.pack(side="left", padx=(0, 8))
        self.open_result_button = ttk.Button(actions, text="打开最近结果", style="Quiet.TButton", command=self.open_last_result, state="disabled")
        self.open_result_button.pack(side="left", padx=(0, 8))
        self.run_button = ttk.Button(actions, text="开始分析", style="Accent.TButton", command=self.run)
        self.run_button.pack(side="left")

    def _refresh_photo_rows(self) -> None:
        selected_count = sum(path is not None for path in self.current)
        self.selection_var.set(f"已选择 {selected_count} / 3 张照片")
        for index, path in enumerate(self.current):
            if path is None:
                self.photo_labels[index].configure(text="尚未选择", foreground=self.COLORS["muted"])
            else:
                self.photo_labels[index].configure(text=Path(path).name, foreground=self.COLORS["text"])

    def pick_region(self, index: int) -> None:
        path = filedialog.askopenfilename(title=f"选择{REGIONS[index]}照片", filetypes=IMAGE_TYPES)
        if path:
            self.current[index] = str(Path(path).resolve())
            self._refresh_photo_rows()

    def pick_current(self) -> None:
        selected = filedialog.askopenfilenames(
            title="选择本次左脸、右脸、下巴各一张（文件名带位置时自动排序）",
            filetypes=IMAGE_TYPES,
        )
        if not selected:
            return
        if len(selected) != 3:
            messagebox.showerror("需要 3 张照片", f"当前选择了 {len(selected)} 张。请按三个视角各选 1 张。", parent=self.root)
            return
        ordered = order_named_views(list(selected))
        self.current = [str(Path(path).resolve()) for path in (ordered or list(selected))]
        self._refresh_photo_rows()
        if ordered is None:
            self.status_var.set("已按文件选择顺序填入三个视角；请核对每张照片所在的行。")

    def clear_current(self) -> None:
        self.current = [None, None, None]
        self._refresh_photo_rows()

    def pick_previous(self) -> None:
        selected = filedialog.askopenfilenames(
            title="从同一个已完成结果目录选择 current_01/02/03_annotated.jpg 三张图",
            initialdir=str(self.baseline_run) if self.baseline_run and self.baseline_run.is_dir() else str(OUTPUTS_DIR),
            filetypes=[("JPEG 图像", "*.jpg *.jpeg")],
        )
        if not selected:
            return
        expected = {"current_01_annotated.jpg", "current_02_annotated.jpg", "current_03_annotated.jpg"}
        selected_paths = [Path(path).resolve() for path in selected]
        if len(selected_paths) != 3 or {path.name for path in selected_paths} != expected:
            messagebox.showerror("需要 3 张已检查结果图", "请选择同一结果目录中的 current_01_annotated.jpg、current_02_annotated.jpg 和 current_03_annotated.jpg。", parent=self.root)
            return
        parents = {path.parent for path in selected_paths}
        if len(parents) != 1:
            messagebox.showerror("基准目录不一致", "三张已检查结果图必须来自同一个结果目录。", parent=self.root)
            return
        baseline_run = parents.pop()
        if not self.is_reviewed_run(baseline_run):
            messagebox.showerror("不是已复核结果", "所选目录没有可复用的已复核框、等级和结果文件。", parent=self.root)
            return
        self.baseline_run = baseline_run
        self._refresh_baseline()

    def _refresh_baseline(self) -> None:
        self.baseline_var.set(self.describe_previous())
        self.baseline_label.configure(fg=self.COLORS["text"] if self.baseline_run else self.COLORS["muted"])

    def clear_previous(self) -> None:
        self.baseline_run = None
        self._refresh_baseline()

    def _python_command(self) -> list[str]:
        python = VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)
        return [str(python), "-X", "utf8", str(SCRIPT)]

    def run(self) -> None:
        if any(path is None for path in self.current):
            missing = [REGIONS[index] for index, path in enumerate(self.current) if path is None]
            messagebox.showerror("照片还没选齐", "请补选以下视角：" + "、".join(missing), parent=self.root)
            return
        command = self._python_command() + ["--current", *[str(path) for path in self.current]]
        if self.baseline_run is not None:
            command.extend(["--baseline-run", str(self.baseline_run)])
        command.extend(["--regions", "左侧视角", "右侧视角", "下巴视角", "--review-boxes"])
        self._set_busy(True, "正在检测本次三张照片；请复核病灶框。通过配准和匹配筛选的对应会自动延续编号。")

        def worker() -> None:
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr or completed.stdout or f"分析脚本退出码 {completed.returncode}")
                self.events.put(("success", completed.stdout.strip()))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(150, self.poll_result)

    def view_history(self) -> None:
        command = self._python_command() + ["--history-only", "--output-dir", str(ROOT / "outputs" / "acne_monitor_history")]
        self._set_busy(True, "正在汇总已保存的随访结果；不会运行痘痘检测或分级模型。")

        def worker() -> None:
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr or completed.stdout or f"历史汇总脚本退出码 {completed.returncode}")
                self.events.put(("history_success", completed.stdout.strip()))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(150, self.poll_result)

    def _set_busy(self, busy: bool, message: str) -> None:
        state = "disabled" if busy else "normal"
        self.run_button.configure(state=state)
        self.history_button.configure(state=state)
        self.status_var.set(message)
        if busy:
            self.progress.pack(fill="x", pady=(7, 0))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def poll_result(self) -> None:
        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(150, self.poll_result)
            return
        self._set_busy(False, "分析失败" if kind == "error" else "已完成")
        if kind == "error":
            messagebox.showerror("分析失败", str(payload), parent=self.root)
            return

        output_path = ""
        for line in str(payload).splitlines():
            if line.startswith("Output: "):
                output_path = line.removeprefix("Output: ").strip()
                break
        if kind == "history_success":
            session_count = next((line.partition(":")[2].strip() for line in str(payload).splitlines() if line.startswith("History sessions:")), "")
            self.status_var.set(f"长期趋势已更新 · {session_count or '时点数量未知'} 个时点")
            image_names = ("longitudinal.png",)
        else:
            self.status_var.set("病灶框复核完成；自动对应结果已生成并打开。")
            image_names = ("overview.png", "longitudinal.png")

        if not output_path:
            return
        result_dir = Path(output_path)
        if kind != "history_success":
            self.last_output = result_dir
            self.open_result_button.configure(state="normal")
            self.sessions_label.configure(text=f"本机已保存 {len([p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]) if OUTPUTS_DIR.is_dir() else 0} 次随访运行")
        missing = [result_dir / name for name in image_names if not (result_dir / name).is_file()]
        if missing:
            messagebox.showerror("结果图片未生成", "未找到结果图片：\n" + "\n".join(str(path) for path in missing), parent=self.root)
            return
        try:
            for image_name in image_names:
                os.startfile(result_dir / image_name)
        except OSError as error:
            self.status_var.set(f"结果已保存，但无法自动打开图片：{error}")
            messagebox.showerror("自动打开失败", f"结果已保存在：\n{result_dir}\n\n{error}", parent=self.root)

    def open_output_folder(self) -> None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(OUTPUTS_DIR)

    def open_last_result(self) -> None:
        if self.last_output and self.last_output.is_dir():
            os.startfile(self.last_output)


if __name__ == "__main__":
    app = tk.Tk()
    AcneMonitorDesktop(app)
    app.mainloop()
