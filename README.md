# 图纸分块定位 API

把大幅图纸切块扫描得到的多张图块，按**已声明的重叠图对**和每对的整数平移搜索窗，
求出各块在整图上的位置。只有所有指定图对同时达标且相对布局唯一时，才返回以
**首图左上角为原点**的拼接坐标。

技术栈：FastAPI + NumPy（Pillow 仅用于 PNG 解码）。提供 Docker Compose 与可交互的
`/docs`。

## 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

生成一套演示图与请求：

```bash
python scripts/make_demo.py demo          # 需要先 pip install -r requirements-dev.txt
docker compose up --build                 # 另一个终端
bash demo/request.sh                      # 期望 status = "aligned"
```

不用 Docker 时：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/demo_client.py --base-url http://localhost:8000
pytest
```

## 接口

`POST /align`，`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `images` | 3～5 张 **8 位灰度 PNG**（PNG 颜色类型 0、非隔行、每边 ≤ 1024 px）。上传顺序即图号 `0..n-1`，图 0 为基准。 |
| `spec` | JSON 字符串，结构见下。 |

`GET /health`、`GET /docs`（Swagger UI）、`GET /redoc`、`GET /openapi.json`。

### spec 结构

```jsonc
{
  "pairs": [
    { "a": 0, "b": 1,
      "window": { "dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2 } },
    { "a": 1, "b": 2, "candidates": [[-450, 450], [-451, 450]] }
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```

* `pairs`：声明哪些图对重叠。每对三选一之外的两种写法**必须且只能**给一个：
  * `window`：闭区间矩形，`(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81`；
  * `candidates`：至多 81 个显式整数平移，自动去重。
* `min_overlap_pixels`：整组共用的最少重叠像素数。
* `max_mean_abs_diff`：整组共用的平均绝对灰度差（MAD）上限，0～255。

### 坐标约定

图对 `(a, b)` 的平移 `(dx, dy)` 表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处：

```
(x_b, y_b) = (x_a + dx, y_a + dy)
```

返回的 `positions[i]` 是第 i 张图左上角在整图坐标中的位置，图 0 恒为 `(0, 0)`。

### 判定逻辑

1. **连通性**：把声明的图对当作无向边，若某图块无法从图 0 到达，立即返回
   `disconnected`，并在 `rescan_image_id` 指出建议补扫的一块（取连通度最低者，平局取小编号）。
2. **图对达标**：对每对枚举搜索窗内的每个整数平移，计算重叠矩形内的 MAD。一个平移被接受需同时满足
   *重叠像素数 ≥ `min_overlap_pixels`* 且 *MAD ≤ `max_mean_abs_diff`*。
   任一对没有任何被接受的平移 → `no_solution`，`failing_pairs` 给出原因
   （`insufficient_overlap` 或 `mean_abs_diff_too_high`）与最佳尝试值。
3. **全局布局**：每对可能有多于一个达标平移。求解器在图对图上建生成树（优先候选少的边），
   枚举树边候选组合，剪掉任何与已放置节点冲突的组合，并在叶子处校验其余冗余图对；
   不同的位置向量即不同的布局。

### 响应

HTTP 状态码只表示请求是否合法；对齐结论在 `status` 字段（均返回 200）：

| `status` | 含义 | 关键字段 |
| --- | --- | --- |
| `aligned` | 所有图对同时达标且布局唯一 | `positions`、`pairs`、`mosaic_bounds`、`layout_unique=true` |
| `disconnected` | 有图块连不到首图 | `unreachable_image_ids`、`rescan_image_id` |
| `no_solution` | 无法同时对齐 | `failing_pairs`（单对不达标）或空（各对单独达标但互相矛盾） |
| `ambiguous` | 多解，位置不唯一 | `uncertain_image_ids`、`alternative_layouts`、`search_truncated` |

`pair_candidate_counts` 给出每个声明图对的达标平移个数，便于诊断阈值是否过松。

错误返回：`400` 图像非法（格式/位深/颜色类型/尺寸/数量/`Content-Type`）或 `spec` 不是 JSON；
`422` `spec` 结构不合法、图号越界、图对重复。

## 实现要点

* **解码**：先读 PNG 的 `IHDR` 校验位深=8、颜色类型=0、非隔行、边长≤1024，再用 Pillow 解码并核对尺寸，
  避免把 16 位或调色板图当作灰度图。
* **打分（降计算量版）**：不依赖 SciPy。核心观察是**部分和下界**——若重叠区共 `n` 像素、已累计绝对差和为 `S`
  （未计算像素的差只可能非负），则最终 MAD 必然 ≥ `S/n`。因此：
  1. 每个候选平移按 32 行的行块逐块累计；一旦 `S/n` 超过 `max_mad` 即可**安全提前拒绝**，剩余行不再比较。
     注意下界必须除以**全重叠像素数 n**，而不是已见像素数（否则后续较小差的行会把全量均值拉回门限内，造成误杀）。
  2. 跑完全部行的候选其 MAD 与旧的逐像素全扫描**逐位一致**，所以达标平移集合、顺序与阈值判定完全不变。
  3. 全图对都不达标时，诊断阶段在“朴素全扫描像素数 × 0.5 × 0.9”的硬预算内，按下界升序（平局保持原窗口顺序）
     精算最有希望的候选，并用严格支配剪枝；预算不足以完成任何整块时，报告能得到的最紧部分和下界及对应重叠像素数。
     精确 MAD 平局时仍取原顺序中最早的尝试。
  达标判定本身不依赖预算、始终精确；预算只约束失败诊断阶段的额外精算。`PairScan.abs_diff_evaluations`
  记录实际逐像素绝对差次数，仅用于自测，不进入响应。
* **实测降载**：题设反相四象限 1024×1024、9×9 窗、最少重叠 1,000,000、MAD 上限 3 的场景下，朴素全扫描需
  84,566,416 次逐像素绝对差且无达标位移；优化后仅 **38,053,884 次（约 45%，减半有余）**，并仍精确报告全窗最低
  MAD 253.007843137255（位于 (-4,-4)，重叠 1,040,400 像素）。复现：`python scripts/bench_alignment.py`。
  带纹理真实图块的 9×9 窗实测约为朴素扫描的 12%、单图对约 14 ms。
* **求解**：见上，`MAX_LAYOUTS` / `MAX_NODES_EXPANDED` 限制最坏工作量；超限时返回
  `ambiguous` 且 `search_truncated=true`，把所有非基准图列为不确定——宁可说不确定，也不谎报唯一。
  预算在尚未找到任何布局时就耗尽的，同样报 `ambiguous` 而不是 `no_solution`，因为此时无法证明无解。

## 目录

```
app/
  main.py       FastAPI 路由、spec 解析与错误映射
  models.py     Pydantic 请求/响应模型与上限常量
  imaging.py    严格 PNG 解码、重叠矩形
  alignment.py  单图对搜索窗打分（部分和下界提前拒绝 + 预算内失败诊断）
  solver.py     全局布局枚举与唯一性判定
  service.py    编排：连通性 → 打分 → 求解
tests/          求解器单测、HTTP 端到端，以及 test_alignment_effort.py
                的达标边界/失败诊断/平局顺序/计算量减半回归
scripts/        make_demo.py 生成演示数据；demo_client.py 直接调用运行中的服务；
                demo_data.py 生成演示图（不依赖 tests/）；
                bench_alignment.py 复现逐像素计算量减半的前后对比
Dockerfile      python:3.11-slim、非 root、HEALTHCHECK
docker-compose.yml  端口 8000、只读根文件系统、健康检查
```

## 已知限制

* 只支持整数像素平移，不处理旋转、比例差或亚像素错位；题目要求同比例、同朝向。
* 判定“达标”只看重叠区灰度 MAD；若重叠区是空白或高度重复纹理，可能出现多解，此时按设计返回
  `ambiguous` 而不是猜测。
* MAD 使用全图块的重叠矩形直接比较，不做事先的亮度/对比度归一化；扫描曝光差异需通过
  `max_mean_abs_diff` 放宽，或由扫描端统一曝光。
