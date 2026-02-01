// 课程章节详细内容 - 基于PDF课件的二级标题

export interface SectionContent {
  id: string;
  title: string;
  subtitle?: string;
  description: string;
  keyPoints: string[];
  image?: string;
  codeExample?: string;
}

export interface Chapter {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  hours: number;
  type: 'theory' | 'practice';
  sections: SectionContent[];
}

export const courseChapters: Chapter[] = [
  {
    id: 'ch1',
    title: '人工智能概述',
    subtitle: 'AI Overview',
    description: '人工智能发展历程、DeepSeek技术突破、AI在海洋学中的应用方向',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch1-1',
        title: '人工智能的发展',
        description: '人工智能发展起步于上世纪40年代，经历多次起伏。21世纪以来人工智能（深度学习）的兴起，很大程度上依赖于算力、算法的提升，而非理论的发展。',
        keyPoints: [
          '1955年，John McCarthy首次提出"人工智能"概念',
          '从手写字体识别到AlphaGo、AlphaFold、DeepSeek等应用',
          'AI方法已证明其效率和能力，影响和改变很多学科',
          'DeepSeek 7天破亿用户，展示颠覆性增长'
        ],
        image: '/images/ai-development.png'
      },
      {
        id: 'ch1-2',
        title: 'DeepSeek的过人之处',
        description: 'DeepSeek打破了美国人工智能算力霸权，证明在相对较低算力条件下也能产生强大的AI。',
        keyPoints: [
          '训练成本对比：560万美元 vs 2.8亿美元',
          '核心公式：AI竞争力 = (算法创新×数据质量)/算力依赖',
          '全球产业格局重构，微软Azure/AWS/英伟达/AMD支持',
          '大语言模型的正确使用：总结知识、编写代码、修改语法'
        ],
        image: '/images/deepseek.png'
      },
      {
        id: 'ch1-3',
        title: '人工智能海洋学的主要应用方向',
        description: 'AI在海洋学中的五大主要应用方向，从数据构建到智能数值模式开发。',
        keyPoints: [
          '海洋大数据构建：深海遥感数据重建、遥感反演',
          '海洋现象的智能识别：中尺度涡、内波等的识别与分类',
          '海洋变量的智能预报：海温、海浪等预测',
          '模式参数的智能计算：参数化方案优化',
          '智能海洋数值模式开发：AI求解动力方程'
        ],
        image: '/images/ai-ocean-apps.png'
      }
    ]
  },
  {
    id: 'ch2',
    title: '海洋大数据简介',
    subtitle: 'Ocean Big Data',
    description: '大数据概况、海洋数据发展历程、数据来源与特征、常用数据平台',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch2-1',
        title: '大数据概况',
        description: '在信息时代背景下，大数据成为各行各业不可忽视的重要资源。海洋大数据具有海量、多样、时变和异构的特点。',
        keyPoints: [
          '"海洋强国"战略与"一带一路"倡议',
          '空天地海大数据总存量达EB级，日增量达PB级',
          '海洋大数据增速40%，超过空天和陆地数据',
          '大数据技术：存储管理、分析挖掘、可视化'
        ],
        image: '/images/big-data.png'
      },
      {
        id: 'ch2-2',
        title: '海洋数据的发展历程',
        description: '现代海洋科学经历三个发展阶段：科学牵引阶段、技术驱动阶段、数据主导阶段。',
        keyPoints: [
          '初步积累阶段（20世纪初）：海洋探测器、测量船',
          '进一步积累阶段（20世纪中）：浮标、遥感卫星、Argo',
          '大量积累阶段（21世纪）：传感器普及、云计算应用',
          '现代物理海洋学之父：Ekman、Rossby、Sverdrup、Stommel、Munk'
        ],
        image: '/images/ocean-data-history.png'
      },
      {
        id: 'ch2-3',
        title: '海洋大数据的5V特征',
        description: '海洋大数据具有Volume（海量性）、Velocity（速度）、Variety（多样性）、Veracity（真实性）、Value（价值）五大特征。',
        keyPoints: [
          'Volume：全球观测网络持续扩充数据量',
          'Velocity：传感器精度提高，采集速度加快',
          'Variety：水文、气象、地质、生物等多领域数据',
          'Value：海洋资源利用、环境保护、灾害预测'
        ],
        image: '/images/5v-characteristics.png'
      },
      {
        id: 'ch2-4',
        title: '常用海洋大数据平台',
        description: '介绍国内外主要的海洋数据获取平台和资源。',
        keyPoints: [
          'Copernicus Marine Service：欧盟最大地球观测计划',
          'NDBC：美国国家数据浮标中心',
          '日本气象厅平台',
          '国家海洋科学数据中心、科学数据银行'
        ],
        image: '/images/data-platforms.png'
      }
    ]
  },
  {
    id: 'ch3',
    title: '神经网络基础',
    subtitle: 'Neural Networks',
    description: '机器学习基础、梯度下降算法、神经网络结构与反向传播',
    hours: 6,
    type: 'theory',
    sections: [
      {
        id: 'ch3-1',
        title: '机器学习基础',
        description: '机器学习是达到人工智能的途径，使机器在没有明确编程的情况下进行"学习"。',
        keyPoints: [
          '监督学习：提供正确训练样本，包括回归和分类',
          '非监督学习：没有具体训练样本，包括聚类',
          '半监督/自监督学习、生成对抗网络',
          '机器学习基本流程：训练→验证→测试'
        ],
        image: '/images/machine-learning.png'
      },
      {
        id: 'ch3-2',
        title: '线性回归与代价函数',
        description: '以海洋热含量(OHC)和海表温度(SST)的线性关系为例，介绍回归问题的基本方法。',
        keyPoints: [
          '模型：OHC = θ₀ + θ₁ × SST',
          '代价函数：均方误差 J(θ)',
          '目标：寻找使J最小化的参数θ',
          '参数空间与变量空间的概念'
        ],
        codeExample: `# 线性回归模型
import numpy as np

# 假设数据
sst = np.array([...])  # 海表温度
ohc = np.array([...])  # 海洋热含量

# 线性模型参数
theta0 = 0.5
theta1 = 2.0

# 预测
ohc_pred = theta0 + theta1 * sst

# 代价函数（均方误差）
def cost_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)`
      },
      {
        id: 'ch3-3',
        title: '梯度下降算法',
        description: '梯度下降是优化神经网络的核心算法，通过迭代更新参数使代价函数最小化。',
        keyPoints: [
          '学习率α的选择：太小收敛慢，太大可能发散',
          '同步更新 vs 异步更新',
          '梯度下降能收敛到局部最小值',
          '接近最优时自动减小步长'
        ],
        codeExample: `# 梯度下降算法
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        # 预测
        predictions = X @ theta
        
        # 计算梯度
        errors = predictions - y
        gradient = (2/m) * X.T @ errors
        
        # 同步更新参数
        theta = theta - alpha * gradient
        
    return theta`
      },
      {
        id: 'ch3-4',
        title: '人工神经网络结构',
        description: '神经网络由输入层、隐藏层、输出层组成，通过激活函数引入非线性。',
        keyPoints: [
          '感知机(Perceptron)：神经网络的起源',
          '神经元、输入层、隐藏层、输出层',
          '深度与宽度的概念',
          '万能近似定理：神经网络能拟合任意连续函数'
        ],
        image: '/images/neural-network-structure.png'
      },
      {
        id: 'ch3-5',
        title: '激活函数',
        description: '激活函数帮助网络学习数据中的复杂模式，引入非线性是神经网络的关键。',
        keyPoints: [
          'Sigmoid：S型曲线，输出0-1',
          'ReLU：线性整流，解决梯度消失问题',
          '为什么需要非线性激活函数？',
          'ReLU vs Sigmoid：收敛速度、计算效率'
        ],
        codeExample: `# 激活函数实现
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)`
      },
      {
        id: 'ch3-6',
        title: '反向传播算法',
        description: '反向传播是训练神经网络的核心算法，通过链式法则计算梯度。',
        keyPoints: [
          '前向传播：计算预测值',
          '反向传播：计算梯度',
          '链式法则的应用',
          '参数更新：梯度下降'
        ],
        codeExample: `# 反向传播示意
# 1. 前向传播
z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)

# 2. 计算误差
delta2 = (y_hat - y) * sigmoid_derivative(z2)

# 3. 反向传播误差
delta1 = (W2.T @ delta2) * sigmoid_derivative(z1)

# 4. 计算梯度
dW2 = delta2 @ a1.T
dW1 = delta1 @ x.T

# 5. 更新参数
W1 = W1 - alpha * dW1
W2 = W2 - alpha * dW2`
      }
    ]
  },
  {
    id: 'ch4',
    title: 'PyTorch基础',
    subtitle: 'PyTorch Basics',
    description: 'Tensor操作、自动微分、数据加载、模型构建与训练',
    hours: 1,
    type: 'practice',
    sections: [
      {
        id: 'ch4-1',
        title: 'Tensor基础操作',
        description: 'PyTorch使用Tensor作为基本数据结构，与NumPy高度兼容但支持GPU加速。',
        keyPoints: [
          'Tensor创建：torch.tensor(), torch.zeros(), torch.randn()',
          'Tensor操作：reshape, transpose, permute',
          'GPU加速：tensor.to(\'cuda\')',
          '与NumPy的相互转换'
        ],
        codeExample: `import torch

# 创建Tensor
x = torch.randn(4, 4)

# 基本操作
x_t = x.t()  # 转置
x_view = x.view(16)  # 变形
x_permute = x.permute(1, 0)  # 维度交换

# 统计操作
mean = torch.mean(x)
std = torch.std(x)

# GPU加速
if torch.cuda.is_available():
    x_gpu = x.cuda()`
      },
      {
        id: 'ch4-2',
        title: '自动微分 Autograd',
        description: 'PyTorch的autograd自动计算梯度，是深度学习训练的核心。',
        keyPoints: [
          'requires_grad=True 追踪计算',
          'backward() 自动计算梯度',
          'grad 属性获取梯度',
          'no_grad() 禁用梯度计算'
        ],
        codeExample: `import torch

# 定义需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 前向计算
y = x ** 2 + 3 * x + 1

# 反向传播
y.backward()

# 获取梯度
print(x.grad)  # dy/dx = 2x + 3 = 7`
      },
      {
        id: 'ch4-3',
        title: 'Dataset与DataLoader',
        description: 'PyTorch提供灵活的数据加载机制，支持批量训练和数据增强。',
        keyPoints: [
          'Dataset类：自定义数据集',
          'DataLoader：批量加载、打乱、多进程',
          'TensorDataset：简单数据包装',
          'batch_size和shuffle参数'
        ],
        codeExample: `from torch.utils.data import Dataset, DataLoader, TensorDataset

# 创建数据集
dataset = TensorDataset(X_train, y_train)

# 创建DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

# 迭代训练
for batch_x, batch_y in dataloader:
    # 训练代码
    ...`
      },
      {
        id: 'ch4-4',
        title: 'nn.Module构建模型',
        description: '使用nn.Module定义神经网络，简洁高效。',
        keyPoints: [
          '继承nn.Module',
          '__init__定义网络层',
          'forward定义前向传播',
          '自动计算反向传播'
        ],
        codeExample: `import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 实例化模型
model = NeuralNetwork()`
      }
    ]
  },
  {
    id: 'ch5',
    title: '深度学习',
    subtitle: 'Deep Learning',
    description: '卷积神经网络CNN、循环神经网络RNN、注意力机制',
    hours: 5,
    type: 'theory',
    sections: [
      {
        id: 'ch5-1',
        title: '卷积神经网络 CNN',
        description: 'CNN是处理图像数据的核心架构，通过卷积操作提取特征。',
        keyPoints: [
          '卷积核（Kernel）与滤波器',
          '共享参数，仅使用局部特征',
          '平移不变性',
          '层次特征提取：像素→边缘→形状→对象'
        ],
        image: '/images/cnn-convolution.png'
      },
      {
        id: 'ch5-2',
        title: '卷积操作的本质',
        description: '卷积操作本质上是对图像进行滤波处理，可以提取边缘、纹理等特征。',
        keyPoints: [
          '卷积核可以实现差分运算',
          '一阶差分：边缘检测',
          '二阶差分：Laplacian算子',
          '海洋图像中的卷积应用'
        ],
        codeExample: `# 卷积操作示例
import torch.nn as nn

# 定义卷积层
conv = nn.Conv2d(
    in_channels=1,   # 输入通道
    out_channels=6,  # 输出通道
    kernel_size=5,   # 卷积核大小
    padding=2        # 填充
)

# 边缘检测卷积核
edge_kernel = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)`
      },
      {
        id: 'ch5-3',
        title: '池化层与批标准化',
        description: '池化层压缩信息，批标准化加速训练。',
        keyPoints: [
          'Max Pooling：保留最显著特征',
          'Average Pooling：平滑特征',
          'Batch Normalization：稳定训练',
          'Dropout：防止过拟合'
        ],
        codeExample: `import torch.nn as nn

# 池化层
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 批标准化
bn = nn.BatchNorm2d(num_features=64)

# Dropout
dropout = nn.Dropout(p=0.5)`
      },
      {
        id: 'ch5-4',
        title: 'LeNet-5 手写字体识别',
        description: 'LeNet是经典的CNN架构，用于MNIST手写字体识别。',
        keyPoints: [
          'LeCun 1998年提出',
          '结构：Conv→Pool→Conv→Pool→FC→FC→Output',
          'ATM机中仍在使用的算法',
          '深度学习三巨头的贡献'
        ],
        image: '/images/lenet.png'
      },
      {
        id: 'ch5-5',
        title: '循环神经网络 RNN',
        description: 'RNN处理序列数据，具有"记忆"能力。',
        keyPoints: [
          '处理时间序列数据',
          '隐含状态传递信息',
          'Encoder-Decoder结构',
          '长期依赖问题'
        ],
        image: '/images/rnn-structure.png'
      },
      {
        id: 'ch5-6',
        title: 'LSTM长短期记忆网络',
        description: 'LSTM通过门控机制解决RNN的长期依赖问题。',
        keyPoints: [
          '遗忘门：丢弃无用信息',
          '输入门：选择新信息',
          '输出门：决定输出',
          '细胞状态：长期记忆'
        ],
        codeExample: `import torch.nn as nn

# LSTM层
lstm = nn.LSTM(
    input_size=10,    # 输入特征数
    hidden_size=64,   # 隐藏层大小
    num_layers=2,     # LSTM层数
    batch_first=True  # 批次维度在前
)

# 前向传播
output, (hidden, cell) = lstm(input_seq)`
      }
    ]
  },
  {
    id: 'ch6',
    title: '海洋回归问题 - 重建',
    subtitle: 'Ocean Regression - Reconstruction',
    description: '深海遥感、U-Net网络、热含量重建、超分辨率重建',
    hours: 2,
    type: 'theory',
    sections: [
      {
        id: 'ch6-1',
        title: '深海遥感简介',
        description: '从表层遥感资料重建海洋内部结构，拓展卫星观测维度。',
        keyPoints: [
          '定义：从表层遥感资料重建海洋内部',
          '目标：提高时空分辨率和覆盖',
          '方法：动力诊断 vs 经验性方法',
          'AI优势：学习表层与内部的映射关系'
        ],
        image: '/images/deep-ocean-remote.png'
      },
      {
        id: 'ch6-2',
        title: 'U-Net架构',
        description: 'U-Net是编码器-解码器结构，广泛应用于图像分割和海洋数据重建。',
        keyPoints: [
          'Contraction Phase：下采样提取特征',
          'Expansion Phase：上采样恢复细节',
          'Skip Connection：保留空间信息',
          '海洋应用：OHC重建、超分辨率'
        ],
        image: '/images/unet-architecture.png'
      },
      {
        id: 'ch6-3',
        title: '海洋热含量OHC重建',
        description: '利用神经网络从SST、SSH等表层数据重建海洋热含量。',
        keyPoints: [
          'OHC是地球系统能量不平衡的最佳表征',
          'OPEN数据集：1993-2020全球OHC',
          '聚类神经网络方法',
          'LSTM时序建模方法'
        ],
        image: '/images/ohc-reconstruction.png'
      },
      {
        id: 'ch6-4',
        title: '超分辨率重建',
        description: '将低分辨率海洋数据重建为高分辨率数据。',
        keyPoints: [
          '1度→0.25度温度产品',
          'CNN超分辨率方法',
          '集成学习方法',
          'eddy-resolved产品构建'
        ],
        image: '/images/super-resolution.png'
      }
    ]
  },
  {
    id: 'ch7',
    title: '海洋时序问题',
    subtitle: 'Ocean Time Series',
    description: '时间序列预测、LSTM应用、ConvLSTM、时空预测',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch7-1',
        title: '时间序列基础',
        description: '时间序列由趋势、季节、循环、不规则波动组成。',
        keyPoints: [
          '长期趋势（Trend）',
          '季节波动（Seasonality）',
          '循环波动（Cyclicity）',
          '不规则波动（Irregularity）'
        ],
        image: '/images/time-series-components.png'
      },
      {
        id: 'ch7-2',
        title: 'AR1模型与海洋记忆',
        description: '海洋具有长期记忆特性，AR1模型可用于分析。',
        keyPoints: [
          'AR1模型：y(t) = α·y(t-1) + ε(t)',
          '海洋记忆时间尺度 τ = -1/ln(α)',
          '相比大气，海洋记忆更长',
          'NARX模型：加入外部输入'
        ],
        codeExample: `# AR1模型
import numpy as np

# AR1参数
alpha = 0.9  # 自相关系数

# 生成AR1序列
y = np.zeros(1000)
y[0] = np.random.randn()
for t in range(1, 1000):
    y[t] = alpha * y[t-1] + np.random.randn()

# 计算记忆时间尺度
tau = -1 / np.log(alpha)`
      },
      {
        id: 'ch7-3',
        title: 'ConvLSTM时空预测',
        description: 'ConvLSTM结合卷积和LSTM，用于时空序列预测。',
        keyPoints: [
          '空间信息：卷积提取',
          '时间信息：LSTM建模',
          'Encoder-Forecaster架构',
          '应用：降水预测、ENSO预报'
        ],
        image: '/images/convlstm.png'
      }
    ]
  },
  {
    id: 'ch8',
    title: '海洋预测 - 注意力机制',
    subtitle: 'Ocean Prediction - Attention',
    description: 'Transformer架构、自注意力机制、时空注意力、波浪预测',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch8-1',
        title: '注意力机制原理',
        description: '注意力机制模仿人类观察方式，为重要信息赋予更大权重。',
        keyPoints: [
          'Query-Key-Value机制',
          '计算输入输出相关性',
          'Softmax归一化权重',
          '加权聚合Value'
        ],
        image: '/images/attention-mechanism.png'
      },
      {
        id: 'ch8-2',
        title: 'Self-Attention自注意力',
        description: '自注意力计算序列内部元素间的相关性。',
        keyPoints: [
          '同一序列的Q、K、V',
          '捕捉长距离依赖',
          '并行计算优势',
          'Transformer的核心'
        ],
        codeExample: `# 自注意力示意
import torch
import torch.nn as nn

# 输入序列
x = torch.randn(batch_size, seq_len, d_model)

# Q, K, V投影
Q = nn.Linear(d_model, d_k)(x)
K = nn.Linear(d_model, d_k)(x)
V = nn.Linear(d_model, d_v)(x)

# 计算注意力分数
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
attn_weights = torch.softmax(scores, dim=-1)

# 加权输出
output = attn_weights @ V`
      },
      {
        id: 'ch8-3',
        title: 'Transformer架构',
        description: '"Attention is All You Need"，Transformer彻底改变了NLP和时序预测。',
        keyPoints: [
          'Multi-Head Attention',
          'Position Encoding',
          'Feed-Forward Networks',
          'Layer Normalization'
        ],
        image: '/images/transformer.png'
      },
      {
        id: 'ch8-4',
        title: 'Vision Transformer (ViT)',
        description: '将Transformer应用于图像问题，在海洋预测中展现强大能力。',
        keyPoints: [
          '图像分块（Patch）',
          '线性嵌入',
          '位置编码',
          '海洋波浪预测应用'
        ],
        image: '/images/vit.png'
      }
    ]
  },
  {
    id: 'ch9',
    title: '海洋识别问题',
    subtitle: 'Ocean Recognition',
    description: '目标检测、语义分割、涡旋识别、R-CNN与YOLO',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch9-1',
        title: '目标检测基础',
        description: '目标检测识别图像中的特定物体并确定位置。',
        keyPoints: [
          '边界框（Bounding Box）',
          '锚框（Anchor Box）',
          '交并比（IoU）',
          '非极大值抑制（NMS）'
        ],
        image: '/images/object-detection.png'
      },
      {
        id: 'ch9-2',
        title: 'R-CNN系列',
        description: 'R-CNN及其改进版本是经典的目标检测方法。',
        keyPoints: [
          'R-CNN：选择性搜索+CNN',
          'Fast R-CNN：RoI池化',
          'Faster R-CNN：RPN网络',
          'Mask R-CNN：实例分割'
        ],
        image: '/images/rcnn.png'
      },
      {
        id: 'ch9-3',
        title: 'YOLO与SSD',
        description: '单阶段检测器，速度快，适合实时应用。',
        keyPoints: [
          'YOLO：You Only Look Once',
          'SSD：Single Shot MultiBox Detector',
          '多尺度特征检测',
          '海洋应用场景'
        ],
        image: '/images/yolo.png'
      },
      {
        id: 'ch9-4',
        title: '中尺度涡识别',
        description: '利用深度学习自动识别海洋中的中尺度涡旋。',
        keyPoints: [
          '涡旋的重要性：热量、动量输运',
          '传统方法：VG算法',
          '深度学习方法：U-Net分割',
          'PSPNet、DeepLabV3+应用'
        ],
        image: '/images/eddy-detection.png'
      }
    ]
  },
  {
    id: 'ch10',
    title: '物理约束神经网络',
    subtitle: 'Physics-Informed Neural Networks',
    description: 'PINN将物理方程嵌入神经网络，提高模型的物理一致性',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch10-1',
        title: 'PINN简介',
        description: '物理信息神经网络将物理知识嵌入AI模型，实现数据+物理双驱动。',
        keyPoints: [
          '纯数据驱动 vs 物理约束',
          '观测偏置、引导偏置、学习偏置',
          'PINN优势：计算快速、物理一致性',
          '适用于不规则边界和复杂几何'
        ],
        image: '/images/pinn-concept.png'
      },
      {
        id: 'ch10-2',
        title: '热传导方程求解',
        description: '以一维热传导方程为例，展示PINN的基本原理。',
        keyPoints: [
          '热传导方程：∂T/∂t = α·∂²T/∂l²',
          '初始条件与边界条件',
          '数值方法求解（差分法）',
          'CFL条件限制'
        ],
        codeExample: `# 热传导方程数值求解
import numpy as np

# 参数设置
alpha = 0.01  # 热传导系数
dt = 0.01     # 时间步长
dl = 0.1      # 空间步长

# 初始条件
T = np.sin(np.pi * l)

# 时间迭代
for n in range(nt):
    T_new = T.copy()
    T_new[1:-1] = T[1:-1] + dt * alpha / dl**2 * (
        T[:-2] - 2*T[1:-1] + T[2:]
    )
    T = T_new`
      },
      {
        id: 'ch10-3',
        title: 'PINN代价函数',
        description: 'PINN的代价函数包含数据项和物理项两部分。',
        keyPoints: [
          'J_data：数据拟合误差',
          'J_phys：物理方程残差',
          'J = J_data + J_phys',
          '配点（Collocation Points）'
        ],
        codeExample: `# PINN代价函数
# 数据约束
J_data = torch.mean((T_pred - T_true)**2)

# 物理约束（热传导方程）
dT_dt = torch.autograd.grad(T_pred, t, 
    create_graph=True)[0]
d2T_dl2 = torch.autograd.grad(dT_dl, l, 
    create_graph=True)[0]
residual = dT_dt - alpha * d2T_dl2
J_phys = torch.mean(residual**2)

# 总代价函数
loss = J_data + weight_phys * J_phys`
      },
      {
        id: 'ch10-4',
        title: 'PINN vs 传统NN',
        description: '对比PINN与传统神经网络在求解PDE时的差异。',
        keyPoints: [
          '传统NN受采样数量限制',
          'PINN可预测未知区域',
          'PINN对噪声更鲁棒',
          '物理一致性保证'
        ],
        image: '/images/pinn-vs-nn.png'
      }
    ]
  }
];

// 编程练习数据
export interface LabExercise {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  topics: string[];
  codeTemplate?: string;
  hints?: string[];
}

export const labExercises: LabExercise[] = [
  // 基础练习
  {
    id: 'L1.0',
    title: 'Python基本操作',
    description: 'Python基础语法、数据类型、列表、字典等基本操作',
    difficulty: 'beginner',
    category: '基础练习',
    topics: ['变量', '数据类型', '列表', '字典'],
    codeTemplate: `# Python基本操作练习

# 1. 基本数据类型
a = 1           # 整数
b = 2.0         # 浮点数
c = complex(3, 2)  # 复数
d = 'Hello'     # 字符串

# 2. 列表操作
my_list = [1, 2, 3, 4, 5]
# TODO: 完成列表的增删改查操作

# 3. 字典操作
my_dict = {'name': 'AI', 'course': 'Oceanography'}
# TODO: 完成字典的增删改查操作`
  },
  {
    id: 'L1.1',
    title: 'NumPy基础练习',
    description: 'NumPy数组创建、切片、统计操作、矩阵运算',
    difficulty: 'beginner',
    category: '基础练习',
    topics: ['ndarray', '切片', '统计', '矩阵运算'],
    codeTemplate: `import numpy as np

# 1. 创建数组
arr = np.arange(10)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# 2. 数组属性
print(arr.shape, arr.dtype, arr.ndim)

# 3. 切片操作
# TODO: 对arr2d进行多种切片操作

# 4. 统计操作
# TODO: 计算mean, std, max, min`
  },
  {
    id: 'L1.2',
    title: '其他工具基础',
    description: 'Matplotlib绘图、SciPy科学计算基础',
    difficulty: 'beginner',
    category: '基础练习',
    topics: ['Matplotlib', 'SciPy', '可视化'],
    codeTemplate: `import matplotlib.pyplot as plt
import numpy as np

# 1. 绘制简单曲线
x = np.linspace(0, 10, 100)
y = np.sin(x)

# TODO: 绘制y=sin(x)曲线，添加标题、标签

# 2. 绘制散点图
# TODO: 生成随机数据并绘制散点图

# 3. 绘制热力图
# TODO: 使用pcolor/imshow绘制二维数据`
  },
  {
    id: 'L1.3',
    title: 'PyTorch基础',
    description: 'Tensor操作、自动微分、GPU加速',
    difficulty: 'beginner',
    category: '基础练习',
    topics: ['Tensor', 'autograd', 'CUDA'],
    codeTemplate: `import torch

# 1. 创建Tensor
x = torch.arange(10)
y = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# 2. Tensor操作
# TODO: 完成reshape, transpose等操作

# 3. 自动微分
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
# TODO: 计算dy/dx`
  },
  {
    id: 'L1.4',
    title: 'NumPy进阶练习',
    description: '高级数组操作、广播机制、性能优化',
    difficulty: 'intermediate',
    category: '基础练习',
    topics: ['广播', '向量化', '性能优化'],
    codeTemplate: `import numpy as np

# 1. 广播机制
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
# TODO: 理解广播结果

# 2. 向量化操作
# 避免使用for循环，使用向量化操作

# 3. 性能对比
import time
# TODO: 对比向量和循环的性能差异`
  },
  // 海洋数据处理
  {
    id: 'L2.1',
    title: 'NumPy地转差分',
    description: '使用NumPy实现海洋地转流的差分计算',
    difficulty: 'intermediate',
    category: '海洋数据处理',
    topics: ['地转流', '差分', '海洋物理'],
    codeTemplate: `import numpy as np

# 地转流公式：
# u = -g/f * ∂η/∂y
# v = g/f * ∂η/∂x

# 参数
g = 9.8  # 重力加速度
f = 1e-4  # 科氏参数

# 海表高度数据（示例）
eta = np.random.randn(100, 100)

# TODO: 使用numpy差分计算地转流u, v
# 提示：使用np.diff或卷积实现差分`
  },
  // 神经网络基础
  {
    id: 'L3.1',
    title: '线性模型及其优化',
    description: '线性回归、代价函数、梯度下降算法实现',
    difficulty: 'intermediate',
    category: '神经网络基础',
    topics: ['线性回归', '梯度下降', '代价函数'],
    codeTemplate: `import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# 初始化参数
theta = np.zeros((2, 1))  # [theta0, theta1]

# TODO: 实现梯度下降算法
# 1. 定义代价函数
# 2. 计算梯度
# 3. 更新参数
# 4. 迭代优化`
  },
  {
    id: 'L4.1',
    title: '手搓NN模型及其优化',
    description: '从零实现神经网络，包括前向传播和反向传播',
    difficulty: 'advanced',
    category: '神经网络基础',
    topics: ['前向传播', '反向传播', '激活函数'],
    codeTemplate: `import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # TODO: 实现前向传播
        pass
    
    def backward(self, X, y, output):
        # TODO: 实现反向传播
        pass`
  },
  {
    id: 'L4.2',
    title: '手搓NN模型 - 向量化',
    description: '神经网络向量优化版本，提升计算效率',
    difficulty: 'advanced',
    category: '神经网络基础',
    topics: ['向量化', '矩阵运算', '性能优化'],
    codeTemplate: `# 向量化版本的神经网络
import numpy as np

# 对比循环版本和向量化版本的性能
# 使用矩阵运算替代for循环

# TODO: 将向量化操作应用到前向传播和反向传播`
  },
  // PyTorch实践
  {
    id: 'L4.1-PyTorch',
    title: 'NN模型用PyTorch',
    description: '使用PyTorch实现相同的神经网络模型',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['nn.Module', '优化器', '训练循环'],
    codeTemplate: `import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # TODO: 定义网络层
        
    def forward(self, x):
        # TODO: 实现前向传播
        pass

# 实例化模型、定义损失函数和优化器
# 编写训练循环`
  },
  {
    id: 'L5.1',
    title: 'PyTorch基础练习',
    description: 'PyTorch核心功能练习，DataLoader使用',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['DataLoader', 'Dataset', '批量训练'],
    codeTemplate: `import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 创建TensorDataset
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)

# 2. 创建DataLoader
# TODO: 设置batch_size和shuffle

# 3. 迭代训练
# TODO: 编写训练循环`
  },
  {
    id: 'L5.2',
    title: 'PyTorch进阶练习',
    description: '高级PyTorch特性，模型保存加载',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['模型保存', 'GPU训练', '学习率调度'],
    codeTemplate: `import torch

# 1. 模型保存和加载
# torch.save(model.state_dict(), 'model.pth')
# model.load_state_dict(torch.load('model.pth'))

# 2. GPU训练
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# 3. 学习率调度
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)`
  },
  // 深度学习应用
  {
    id: 'LeNet',
    title: 'LeNet手写字体识别',
    description: '实现LeNet-5网络，MNIST数据集分类',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['CNN', 'LeNet', 'MNIST', '分类'],
    codeTemplate: `import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # TODO: 定义卷积层和全连接层
        # Conv1: 1->6 channels, kernel=5
        # Pool1: 2x2
        # Conv2: 6->16 channels, kernel=5
        # Pool2: 2x2
        # FC layers
        
    def forward(self, x):
        # TODO: 实现前向传播
        pass`
  },
  // 新增：U-Net练习
  {
    id: 'L7.1',
    title: 'U-Net海洋数据重建',
    description: '使用U-Net网络进行海洋热含量(OHC)重建',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['U-Net', 'OHC重建', '深海遥感'],
    codeTemplate: `import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder (下采样)
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 2, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)
        
        # TODO: 完成down2, down3
        
        # Decoder (上采样)
        # TODO: 定义up2, up1
        
        # Skip connection处理
        # TODO: 定义skip connection后的卷积
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        # TODO: 完成encoder
        
        # Decoder with skip connections
        # TODO: 完成decoder
        
        return output`
  },
  // 新增：LSTM练习
  {
    id: 'L8.1',
    title: 'LSTM径流预测',
    description: '使用LSTM进行时间序列预测，预测未来径流',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['LSTM', '时间序列', '预测'],
    codeTemplate: `import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TODO: 定义LSTM层
        self.lstm = nn.LSTM(...)
        
        # TODO: 定义输出层
        self.fc = nn.Linear(...)
        
    def forward(self, x):
        # TODO: 初始化h0, c0
        # TODO: LSTM前向传播
        # TODO: 全连接输出
        pass`
  },
  // 新增：PINN练习
  {
    id: 'L11.1',
    title: 'PINN求解热传导方程',
    description: '使用物理信息神经网络求解一维热传导方程',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['PINN', 'PDE', '物理约束'],
    codeTemplate: `import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 定义神经网络层
        self.net = nn.Sequential(
            nn.Linear(2, 50),  # 输入: (t, x)
            nn.Sigmoid(),
            nn.Linear(50, 50),
            nn.Sigmoid(),
            nn.Linear(50, 1),  # 输出: T
        )
        
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        return self.net(inputs)
    
    def physics_loss(self, t, x):
        # TODO: 计算物理方程残差
        # 热传导方程: dT/dt = alpha * d2T/dx2
        pass

# TODO: 定义训练循环，同时优化数据损失和物理损失`
  }
];
