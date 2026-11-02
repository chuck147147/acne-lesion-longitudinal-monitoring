# 病灶级痤疮长期监测

一个本地运行的研究原型：使用 YOLO 检测痤疮候选框，再对每个候选病灶分类；对同一人的多次随访照片按面部锚点配准，输出逐病灶结果、分区统计、对比图和数值摘要。

> 本项目的输出是模型预测和工程原型统计，不是临床诊断或经验证的整脸痤疮严重度量表。请勿据此自行诊断或调整治疗。

## 三张图随访

在 Windows 上双击 [`start_acne_monitor.pyw`](start_acne_monitor.pyw) 打开桌面工作台，可分别为左脸、右脸、下巴选择照片，也可一次选择三张。程序默认复用最近一次已完成人工复核的结果作为基准；也可选择该结果目录中的 `current_01_annotated.jpg` 至 `current_03_annotated.jpg` 切换基准。基准直接读取已复核的框和等级，不再运行痘痘检测或分级模型。人工修正窗口只包含本次三张新照片，基准照片及标注显示在旁边供参考。每次复核完成会更新全时点趋势；“查看长期趋势”按钮也可单独打开历史趋势，不重跑模型。首次使用时可清除基准，只生成当前基线。当前桌面入口仍依赖项目内已安装的 Python 运行环境；独立 `.exe` 尚未打包。

模型检测本次照片后会打开逐框复核工具：只需检查本次三张，可补画漏检、删除误检、重画错误框和修正类别；基准照片仅作为带有已复核标注的参考。未手动指定类别的补画或重画框会重新分类。逐框复核完成后，程序会使用面部配准、位置、框大小和局部外观特征自动建立一对一对应，并延续匹配候选的稳定 `track_id`。自动匹配可能受光线、姿势和取景差异影响；未匹配病灶保留独立编号，不自动判为新发或消退。编号及逐时点观察保存在本机 `outputs/acne_monitor_history/lesion_tracks.csv`。

检测器遇到面积较大的框时，会对该区域局部放大约 3 倍再次检测；若找到多个较小的病灶候选，会用小框替换大框。拆分证据不足时会在复核列表标记“**大框需拆分**”，供人工重画或补框。该步骤使用现有检测器做局部复检，不是像素级分割，也不会通过 NMS 合并相邻的病灶框。

命令行监测使用独立运行依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-monitor.txt
```

非 Windows 系统可直接运行 Python 脚本；`.pyw` 图形入口仅面向 Windows。

如需配置本机默认基线，把 [`monitor.local.example.json`](monitor.local.example.json) 复制为 `monitor.local.json`，再填入上次已复核的结果目录。未配置时会自动选择最近一次已复核的结果。`monitor.local.json` 已加入忽略规则，不会随仓库提交。

同一流程也可从命令行运行：

```powershell
.\.venv\Scripts\python.exe scripts\monitor_three_photos.py `
  --current <本次左侧图> <本次右侧图> <本次下巴图> `
  --baseline-run <上次已复核结果目录> `
  --review-boxes
```

输出保存到 `outputs/acne_monitor_runs/<时间戳>/`，包含：

- `current_01_annotated.jpg` 至 `current_03_annotated.jpg`：本次逐框圈划图。
- `comparison_01.jpg` 至 `comparison_03.jpg`：同视角前后并排对比图。
- `overview.png`：三视角合成总览图；下巴按眼睛和下巴锚点自动摆正，每区图下列出前后分级、检测框数量和变化，底部汇总三视角结果。
- `lesions.csv`：逐病灶类别、置信度、区域和时间状态。
- `candidate_tracks.csv`：结合面部配准、框大小、分级和局部纹理相似度的一对一自动对应、匹配指标和稳定编号。
- `track_observations.csv`、`track_summary.csv`：本机稳定病灶编号及逐次观察摘要。
- `detection_review/`：本次复核工作副本，包含可编辑框表和图片。
- `detection_review/images/previous_01.jpg` 至 `previous_03.jpg`：只读基准参考图，已绘制已复核框；编辑器只开放本次三张照片。
- `region_summary.csv`：每个视角的分型计数和前后变化量。
- `report.md`：图像链接、分区统计和整组三视角数值结论。
- `result.json`：供后续工具读取的机器可读结果。
- `longitudinal.png`、`longitudinal.csv`：该运行时点对应的全历史趋势图和逐时点统计；汇总副本同时保存在 `outputs/acne_monitor_history/`。

长期趋势按照片路径中的日期和时段排列；同一组三张原图会去重，不同照片组会保留为独立时点，因此同一天的不同拍摄批次不会合并。若项目里已有人工确认的历史标签摘要，也会自动纳入活动框历史；早期只有部分视角的记录会注明照片数，不伪造缺失区域的分数。近期标准三视角结果另画分区负担趋势。图中空心点表示未复核或状态未知。趋势覆盖全部可读取的历史结果；逐颗病灶通过相邻时点自动配对延续稳定编号，并累计观察历史。自动配对可能误配；未匹配和本次未检出的病灶不会自动标为消退。

只更新或查看已有记录的趋势，可运行：

```powershell
.\.venv\Scripts\python.exe scripts\monitor_three_photos.py --history-only
```

报告和总览图对每个视角分别给出 0*/I/II/III/IV 数量、活动病灶框数、炎症性框数、结节数和负担分。侧脸配准使用相机侧的眼部锚点和下颌线轮廓，并剔除落在画面外的预测点；左右脸候选再结合配准后的局部颜色/纹理、脸部位置、框大小和类别。当多个候选难以区分时，程序会保留未对应框。自动匹配使用的相似度不是病灶同一性的概率；光线、姿势和取景差异仍可能造成错配。负担分是项目原型：

```text
I × 1 + II × 2 + III × 2 + IV × 4
```

II 丘疹和 III 脓疱同权；`0*` 痘印/瘢痕候选不计入活动病灶。负担分权重尚未经过临床校准。三张不同视角可能拍到同一颗痘，因此三视角计数之和是检测框合计，不等于全脸去重后的唯一病灶数。

跨日期对应由现有候选算法自动建立。候选排序分不是匹配概率；自动错配可能让稳定编号接错，未配对框不会自动称为新发或消退。配准失败时仍输出图像和分区计数，但该视角不会自动延续逐颗编号。

详细口径见 [`docs/三张图自动监测输出规范.md`](docs/三张图自动监测输出规范.md) 和 [`docs/后续随访默认流程.md`](docs/后续随访默认流程.md)。

## 模型类别

现有逐框分类标签为项目操作类别：

| 标签 | 类别 | 活动病灶统计 |
|---|---|---|
| 0* | 痘印或瘢痕候选，无可见活动性皮损 | 排除 |
| I | 粉刺 | 纳入 |
| II | 炎性丘疹 | 纳入 |
| III | 脓疱 | 纳入 |
| IV | 结节或囊肿 | 纳入 |

这些是逐框模型类别，不等于临床整脸分级。分类置信度是模型分数，不代表模型正确概率。

## 本地模型与依赖

照片、训练集和训练得到的模型权重不随仓库发布。三图监测脚本默认读取以下本地检查点；首次使用前需自行准备有权使用的兼容权重，具体接口和放置方式见[模型权重准备说明](docs/模型权重与依赖准备.md)：

- 检测器：`runs/detect/acne04_yolo11n_fold0/weights/best.pt`
- 逐框分类器：`runs/classify/acne_box_severity_yolo11n_v2_finetune/weights/best.pt`
- 面部关键点模型：`weights/face_landmarker.task`

三图推理依赖见 `requirements-monitor.txt`。训练依赖见 `requirements-training.txt`，其中包含 CUDA 版本的 PyTorch 配置；其他操作系统或无 CUDA 环境需按本机平台调整 PyTorch。仓库提供 ACNE04 单类检测器的训练脚本；三图完整运行还需要本地病灶分类器和面部关键点权重。病灶分类器的训练流程尚未纳入当前公开候选版本。`requirements-tracking.txt` 列出配准核心的独立依赖。

命令行支持通过 `--detector`、`--classifier` 和 `--landmarker` 指定本地模型路径。
单图检测器预览也可用 `scripts/predict_acne04.py --image <图片路径>`。

## GitHub 与隐私

`data/`、`outputs/` 和 `reports/` 可能包含面部照片、标注或个人随访结果，已加入 `.gitignore`。本机基线配置、模型检查点和运行产物也不会纳入版本控制。公开仓库仅应包含代码、通用文档和经过授权且去标识化的示例；提交前请检查 `git status` 和待提交文件列表。

发布前逐项核对 [`GITHUB_RELEASE_CHECKLIST.md`](GITHUB_RELEASE_CHECKLIST.md)。
更多数据与问题报告约定见 [`SECURITY.md`](SECURITY.md)。
贡献代码和提交问题前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 许可证

仓库中的项目代码按 [MIT 许可证](LICENSE)发布。该许可证不覆盖个人照片、第三方数据集、外部模型或本地训练得到的权重；这些资源需分别遵守其所有者和来源的授权条件。

## 目录

- `scripts/`：推理、标注、训练、评估和纵向配准脚本；`face_tracking_core.py` 提供三图监测使用的通用配准与候选匹配逻辑，`lesion_track_store.py` 保存逐颗长期编号。
- `docs/`：数据标注口径、评估方案及运行说明。
- `start_*.pyw`：Windows 图形入口。
- `requirements-*.txt`：训练和面部追踪依赖。
