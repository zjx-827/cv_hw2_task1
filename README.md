## 代码说明

这个代码是一个简单的图像分类模型训练和可视化的示例。它使用了预训练的ResNet模型来进行图像分类任务，并通过TensorBoard记录训练过程中的损失和准确率。

### 文件结构

- `train_model.py`: 包含了训练模型的代码，包括数据加载、模型训练、损失计算等功能。
- `initialize_model.py`: 包含了初始化模型的代码，根据选择的模型名称和参数进行模型初始化。
- `README.md`: 项目说明文档。

### 使用方法

1. 在`train_model.py`中设置数据目录`data_dir`和其他参数。
2. 运行`train_model.py`来训练模型并保存最佳模型权重。
3. 运行`visualize_model.py`来查看模型在测试集上的分类效果。

### 数据集

数据集使用的是CUB-200-2011，包含了200种鸟类别的图像数据。训练集和测试集分别存储在`images/train/`和`images/test/`目录下。

### 模型

模型选择了ResNet18作为基础模型，可以通过设置`model_name`参数来选择其他模型。在`initialize_model.py`中可以设置是否使用预训练模型和是否冻结特征提取层。

### 可视化

模型训练完成后，可以通过`visualize_model.py`来查看模型在测试集上的分类效果。可以设置`num_images`参数来控制展示的图像数量。

### 其他

- 使用TensorBoard记录训练过程中的损失和准确率。
- 通过`set_parameter_requires_grad`函数控制是否冻结参数。

## 运行环境

- Python 3.x
- PyTorch
- torchvision
- TensorBoard

## 参考链接

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [CUB-200-2011数据集](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
