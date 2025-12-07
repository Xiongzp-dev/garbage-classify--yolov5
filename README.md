# 基于改进 YOLOv5s 的智能垃圾分类系统

> 《人工神经网络和深度学习》课程设计项目  
> 论文题目：**《基于改进YOLOv5算法的智能垃圾分类系统》**

---

## 1. 项目简介

本项目基于 Ultralytics 开源 YOLOv5 框架，选用 **YOLOv5s** 作为基础网络，针对封闭式智能垃圾桶场景进行改进与二次开发，实现对四类生活垃圾的实时检测与分类。

- 场景：封闭式智能垃圾桶，统一绿色托盘背景 + LED 补光  
- 任务：单目标垃圾检测与四类分类  
- 垃圾类别：
  - `recyclable`：可回收垃圾（纸杯、塑料瓶等）
  - `harmful`：有害垃圾（电池、药片等）
  - `kitchen`：厨余垃圾（胡萝卜、小土豆、白萝卜等）
  - `other`：其他垃圾（瓷片、砖块、鹅卵石等）

在自建数据集上，改进 YOLOv5s 模型在验证集上取得了大约：

- **Precision：0.957**
- **Recall：0.937**
- **mAP@0.5：0.963**
- **mAP@0.5–0.95：0.898**

---

## 2. 算法与训练策略改进

项目的重点是对 YOLOv5s 在小规模、统一成像环境的数据集上的**针对性改进**，主要包括：

1. **锚框自适应聚类（Anchor Clustering）**
   - 使用自建垃圾数据集的标注框做 **K-means++ 聚类**，获得 9 个锚框。
   - 将 `models/yolov5s.yaml` 中的默认 `anchors` 替换为聚类结果，更适配垃圾目标的尺度与长宽比。

2. **损失函数权重与超参数调整**
   - 在 `hyp.garbage.yaml` 中：
     - `box`（边界框回归损失权重）从 `0.05` 提高到 `0.07`
     - `obj`（目标置信度损失权重）从 `1.0` 降到 `0.8`
   - 在统一绿色背景下更强调目标位置/尺度学习，减弱对背景的过拟合。

3. **分阶段数据增强策略**
   - 使用 YOLOv5 内置数据增强并自定义超参数：
     - Mosaic 拼接、多尺度缩放 `scale`、随机平移 `translate`
     - HSV 颜色扰动 `hsv_h/s/v`、随机翻转、Mixup 等
   - 在 `train.py` 中实现 **“前期强增强、后期弱增强”**：
     - 前 80% epoch：`dataset.mosaic = True`，保留合适的 `dataset.mixup`
     - 后 20% epoch：关闭 Mosaic 与 Mixup，仅保留轻量颜色/尺度增强。

4. **类别不平衡处理与训练策略**
   - 启用 YOLOv5 的 **类别加权图像采样**：
     - 通过训练命令中的 `--image-weights` 调用 `labels_to_image_weights` 函数，
       提升小样本类别（如有害垃圾）的采样频率。
   - 训练命令中加入：
     - **余弦退火学习率**：`--cos-lr`
     - **标签平滑**：`--label-smoothing 0.02`
   - 共同作用下，改进了收敛稳定性和泛化能力。

---

## 3. 项目结构（示例）

实际目录以仓库为准，典型结构如下：

```text
.
├── data/
│   ├── garbage.yaml              # 数据集配置（类别、路径）
│   └── hyps/
│       └── hyp.garbage.yaml      # 自定义超参数（损失权重 & 数据增强）
├── models/
│   └── yolov5s.yaml              # 修改后的 YOLOv5s 结构与锚框
├── utils/                        # YOLOv5 工具函数
├── train.py                      # 训练脚本（含分阶段数据增强）
├── detect.py                     # 推理脚本（PC 端测试）
├── export.py                     # 导出 ONNX / TorchScript
├── README.md                     # 本文件
└── ...
```
---

## 4. 环境配置

### 4.1 训练环境（示例）

- 平台：云端 GPU（如 AutoDL 等）  
- GPU：NVIDIA RTX 系列显卡  
- 系统：Ubuntu 20.04 / 22.04  
- CUDA：与显卡/驱动匹配的版本  
- Python：3.8 ~ 3.12  
- PyTorch：2.x  
- YOLOv5：基于 Ultralytics 官方仓库修改  

### 4.2 本地配置步骤

```bash
# 克隆仓库
git clone https://github.com/Xiongzp-dev/garbage-classify--yolov5.git
cd garbage-yolov5

# 安装依赖
pip install -r requirements.txt
```
---

## 5. 数据集说明

- 数据总量：约 **2100 张** 单目标垃圾图片  
- 成像环境：统一绿色托盘背景 + LED 补光，单次只落入一件垃圾  
- 类别：`recyclable`、`harmful`、`kitchen`、`other` 共 4 类  
- 划分比例：训练集 : 验证集 : 测试集 ≈ **8 : 1 : 1**  
- 标注格式：标准 YOLO 格式（`class cx cy w h`，均为归一化坐标）  

`coco128.yaml` 示例：

```yaml
train: path/to/train/images
val:   path/to/val/images
test:  path/to/test/images

nc: 4
names: [recyclable, harmful, kitchen, other]
```
-出于体积与隐私考虑，仓库中并没有附带完整数据集，可根据论文说明自行采集或联系作者获取。

---

## 6. 训练改进后的 YOLOv5s 模型

在项目根目录下执行以下命令即可训练：

```bash
python train.py \
  --weights yolov5s.pt \
  --cfg models/yolov5s.yaml \
  --data data/garbage.yaml \
  --hyp data/hyps/hyp.garbage.yaml \
  --img 640 \
  --epochs 141 \
  --batch-size 8 \
  --image-weights \
  --cos-lr \
  --label-smoothing 0.02 \
  --project runs/train \
  --name garbage_yolov5s_improved
```
---

## 7. 实验结果概览

在自建垃圾数据集验证集上，改进 YOLOv5s 模型取得了约：

| 指标         | 数值   |
| ------------ | ------ |
| Precision    | 0.957  |
| Recall       | 0.937  |
| mAP@0.5      | 0.963  |
| mAP@0.5–0.95 | 0.898  |

- 训练与验证的 `box_loss`、`cls_loss`、`dfl_loss` 曲线整体平滑下降，无明显过拟合。  
- mAP 曲线在 40–60 epoch 区间达到高值并趋于稳定。  
- 在实际测试中，可回收物与其他垃圾检测置信度普遍在 0.97 以上，有害垃圾中白色药片与瓷片、橙色电池与胡萝卜极少数情况下存在混淆。  

---

## 8. 许可证（示例）

可根据实际情况选择开源协议，这里示例为 MIT：

```text
This project is licensed under the MIT License.
See the LICENSE file for details.
