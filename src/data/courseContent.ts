// 课程章节详细内容 - 基于PDF课件的二级标题

// 获取正确的图片路径（适配 GitHub Pages 子路径）
function getImagePath(path: string): string {
  const base = import.meta.env.BASE_URL
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  return `${base}${cleanPath}`
}

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
  /** 为 true 时仅展示章节与小节标题，不展示详细内容 */
  contentHidden?: boolean;
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
        image: getImagePath('/images/ai-development.png')
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
        image: getImagePath('/images/deepseek.png')
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
        image: getImagePath('/images/ai-ocean-apps.png')
      }
    ]
  },
  // 第二章及之后：保留标题，隐去内容
  {
    id: 'ch2',
    title: '海洋大数据简介',
    subtitle: 'Ocean Big Data',
    description: '大数据概况、发展历程、定义与特征、数据来源以及常用平台',
    hours: 1,
    type: 'theory',
    sections: [
      {
        id: 'ch2-1',
        title: '2.1 大数据概况',
        description: '在信息时代背景下，大数据已成为关键生产要素。海洋大数据具有海量、多样、时变和异构的特点。',
        keyPoints: [
          '国家“海洋强国”战略推动海洋数据基础能力建设',
          '海洋数据体量持续增长，正从“小数据”走向“大数据”',
          '观测网络、遥感技术与算力进步共同驱动数据爆发',
          '海洋大数据核心链路：采集、管理、分析挖掘与可视化'
        ],
        image: getImagePath('/images/big-data.png')
      },
      {
        id: 'ch2-2',
        title: '2.2 海洋大数据发展历程',
        description: '海洋数据发展经历“初步积累—进一步积累—大量积累”三个阶段，并推动现代海洋科学从技术驱动逐步迈向数据主导。',
        keyPoints: [
          '20世纪初：以海洋探测器与测量船为主，数据获取能力有限',
          '20世纪中后期：浮标与卫星遥感快速发展，观测范围显著扩大',
          '21世纪以来：Argo与多源传感器网络推动海量数据累积',
          '从科学牵引到数据主导，海洋研究范式发生结构性变化'
        ],
        image: getImagePath('/images/ocean-data-history.png')
      },
      {
        id: 'ch2-3',
        title: '2.3 海洋大数据定义与5V特征',
        description: '海洋大数据是海洋环境、生态、气候、地质与人类活动等多领域产生的海量数据集合，具有典型的5V特征。',
        keyPoints: [
          'Volume（海量性）：全球观测网络持续扩展，数据规模快速增加',
          'Velocity（速度）：采集频率提升，实时与准实时处理需求增强',
          'Variety（多样性）：水文、气象、地质、生物等多模态异构数据并存',
          'Veracity（真实性）：质量控制决定数据可信度，降低噪声与误差影响',
          'Value（价值性）：数据挖掘支撑资源利用、风险预警与决策优化'
        ],
        image: getImagePath('/images/5v-characteristics.png')
      },
      {
        id: 'ch2-4',
        title: '2.4 海洋大数据来源',
        description: '海洋大数据主要来源于实测、遥感和模式模拟三类数据体系。三者互补融合，构成现代海洋数据基础。',
        keyPoints: [
          '实测数据：来自船舶、浮标、潜标等平台，精度高但覆盖受限',
          '遥感数据：覆盖范围广、更新快，适合大尺度连续观测',
          '模式数据：通过数值模拟补足空时缺测，支撑预测分析',
          '融合应用：多源协同提升海洋状态认知与业务化能力'
        ],
        image: getImagePath('/images/data-sources.png')
      },
      {
        id: 'ch2-5',
        title: '2.5 常用海洋大数据平台',
        description: '介绍课程中常用的国内外海洋数据平台，为后续数据下载、处理与科研实践提供入口。',
        keyPoints: [
          'Copernicus Marine Service：全球海洋产品丰富，支持在线检索与下载',
          'NDBC：提供海洋浮标与海气要素历史与实时观测数据',
          '日本气象厅平台：整合气象与海洋业务化数据产品',
          '国内平台：国家海洋科学数据中心等提供本土化数据资源'
        ],
        image: getImagePath('/images/data-platforms.png')
      }
    ]
  },
  {
    id: 'ch3',
    title: '神经网络基础',
    subtitle: 'Neural Networks',
    description: '机器学习基础、线性回归与代价函数、梯度下降、神经网络结构、激活函数与反向传播',
    hours: 6,
    type: 'theory',
    sections: [
      {
        id: 'ch3-1',
        title: '机器学习基础',
        description: '机器学习是实现人工智能的重要途径，让模型在没有显式规则编程的情况下，从数据中学习规律并提升表现。',
        keyPoints: [
          '监督学习：基于已标注样本，典型任务包括回归与分类',
          '非监督学习：无标签数据中发现结构，典型任务包括聚类',
          '半监督/自监督学习：结合少量标签与大量无标签数据提升效果',
          '基本流程：训练（Training）→ 验证（Validation）→ 测试/推理（Testing/Inference）'
        ],
        image: getImagePath('/images/ch3-ml-workflow.png')
      },
      {
        id: 'ch3-2',
        title: '线性回归与代价函数',
        description: '以海洋热含量（OHC）与海表温度（SST）的关系为例，介绍回归建模与代价函数的基本思想。',
        keyPoints: [
          '模型：OHC = θ₀ + θ₁ × SST',
          '代价函数：均方误差 MSE，对应 J(θ)',
          '优化目标：寻找使 J 最小的参数组合 θ',
          '参数空间中最优点对应变量空间中的最优拟合关系'
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
    return np.mean((y_true - y_pred) ** 2)`,
        image: getImagePath('/images/ch3-linear-regression.png')
      },
      {
        id: 'ch3-3',
        title: '梯度下降算法',
        description: '梯度下降是模型优化的核心方法，通过沿着负梯度方向迭代更新参数，使代价函数逐步减小。',
        keyPoints: [
          '学习率 α 影响收敛速度与稳定性：过小收敛慢，过大可能震荡或发散',
          '参数更新应采用同步更新，避免异步更新引入偏差',
          '复杂损失面中可能收敛到局部极小点',
          '接近最优区域时，梯度变小，更新步长会自然减小'
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
        
    return theta`,
        image: getImagePath('/images/ch3-gradient-descent.png')
      },
      {
        id: 'ch3-4',
        title: '人工神经网络结构',
        description: '神经网络由输入层、隐藏层、输出层组成，通过激活函数引入非线性。',
        keyPoints: [
          '感知机(Perceptron)：神经网络的起源',
          '核心结构：神经元、输入层、隐藏层、输出层',
          '深度与宽度的概念',
          '万能近似定理：神经网络能拟合任意连续函数'
        ],
        image: getImagePath('/images/ch3-neural-network-architecture.png')
      },
      {
        id: 'ch3-5',
        title: '激活函数',
        description: '激活函数为网络引入非线性表达能力，是神经网络拟合复杂模式的关键组件。',
        keyPoints: [
          'Sigmoid：S 形映射，输出范围为 0 到 1',
          'ReLU：计算高效，在深层网络中表现稳定',
          '为什么需要非线性激活函数？',
          '常见对比：ReLU 与 Sigmoid 在收敛速度和梯度传播上的差异'
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
    return (x > 0).astype(float)`,
        image: getImagePath('/images/ch3-activation-functions.png')
      },
      {
        id: 'ch3-6',
        title: '反向传播算法',
        description: '反向传播是神经网络训练的核心步骤，利用链式法则从输出层向前逐层计算梯度并更新参数。',
        keyPoints: [
          '前向传播：计算预测值',
          '反向传播：逐层求导并传递误差信号',
          '链式法则：将复杂网络拆解为可计算的局部梯度',
          '参数更新：结合梯度下降完成权重与偏置迭代'
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
W2 = W2 - alpha * dW2`,
        image: getImagePath('/images/ch3-backpropagation.png')
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
    contentHidden: true,
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
    contentHidden: true,
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
        image: getImagePath('/images/cnn-convolution.png')
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
        image: getImagePath('/images/lenet.png')
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
        image: getImagePath('/images/rnn-structure.png')
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
    contentHidden: true,
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
        image: getImagePath('/images/deep-ocean-remote.png')
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
        image: getImagePath('/images/unet-architecture.png')
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
        image: getImagePath('/images/ohc-reconstruction.png')
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
        image: getImagePath('/images/super-resolution.png')
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
    contentHidden: true,
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
        image: getImagePath('/images/time-series-components.png')
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
        image: getImagePath('/images/convlstm.png')
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
    contentHidden: true,
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
        image: getImagePath('/images/attention-mechanism.png')
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
        image: getImagePath('/images/transformer.png')
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
        image: getImagePath('/images/vit.png')
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
    contentHidden: true,
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
        image: getImagePath('/images/object-detection.png')
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
        image: getImagePath('/images/rcnn.png')
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
        image: getImagePath('/images/yolo.png')
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
        image: getImagePath('/images/eddy-detection.png')
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
    contentHidden: true,
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
        image: getImagePath('/images/pinn-concept.png')
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
        image: getImagePath('/images/pinn-vs-nn.png')
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
  /** 为 true 时仅展示标题，不展示描述与代码模板 */
  hidden?: boolean;
}

// 基础练习：来自 Student_Notebook.ipynb（AI 海洋学课前热身 50 Quizzes）
const notebookBasicTemplate = `# AI 海洋学：课前热身编程练习 (50 Quizzes)
# 本练习旨在帮助你快速复习 Python、NumPy 和 Matplotlib 的核心操作，
# 并掌握向量化编程思维，为后续运用深度学习模型处理复杂的海洋与气候数据打下基础。
# 请在带有 # TODO: 的注释下方编写你的代码。

## 单元一：Python 基本操作 (Quiz 1 - 10)
# 本单元复习基础的数据结构、循环控制与函数封装。

# Quiz 1: 打印一条欢迎信息："欢迎来到 AI 海洋学课程！"
# TODO:

# Quiz 2: 创建一个包含三个测站名称的列表，命名为 stations，元素为 'St_A', 'St_B', 'St_C'。
# TODO:

# Quiz 3: 将"嘉庚"号（'Jiageng'）作为移动测站，追加到 stations 列表的末尾。
# TODO:

# Quiz 4: 创建一个字典 station_coords，键为测站名 'St_A' 和 'St_B'，值为包含经纬度的元组，例如 (113.5, 22.1) 和 (114.0, 21.8) (珠江口附近坐标)。
# TODO:

# Quiz 5: 从字典 station_coords 中提取 'St_A' 的坐标并打印。
# TODO:

# Quiz 6: 使用 for 循环遍历 stations 列表，打印每个测站的名称。
# TODO:

# Quiz 7: 编写一个函数 celsius_to_kelvin(c)，将摄氏度转换为开尔文温度 (K = C + 273.15)。
# TODO:

# Quiz 8: 定义一个包含三个海表温度(SST)的列表 sst_c = [24.5, 28.1, 15.0]，使用 for 循环和上一步的函数，将其全部转换为开尔文并存入新列表 sst_k。
# TODO:

# Quiz 9: 使用列表推导式 (List Comprehension)，从 sst_c 中筛选出大于 25°C 的温度值。
# TODO:

# Quiz 10: 使用 f-string 格式化字符串，打印输出："测站 St_A 的当前水温为 24.5 °C"。
# TODO:

## 单元二：NumPy 基础操作 (Quiz 11 - 25)
# 本单元重点在于多维数组的创建、索引、切片及基本运算。

# Quiz 11: 导入 numpy 库，并简写为 np。
# TODO:

# Quiz 12: 创建一个包含 5 个不同深度盐度值 (如 33.1, 33.5, 34.0, 34.2, 34.5) 的 1D NumPy 数组 salinity。
# TODO:

# Quiz 13: 创建一个 3x3 的 2D NumPy 数组 sst_grid，模拟一小块海域的网格化海表温度。
# TODO:

# Quiz 14: 创建一个 10x10 的全 0 数组 depth_mask，数据类型指定为 float32。
# TODO:

# Quiz 15: 创建一个 5x5 的全 1 数组 land_mask，数据类型指定为 int8。
# TODO:

# Quiz 16: 使用 np.arange 创建一个水深序列 depths，从 0 开始，到 1000 结束（不包含1000），步长为 50。
# TODO:

# Quiz 17: 使用 np.linspace 创建一个纬度序列 lats，在 10 到 20 之间均匀分布 50 个点。
# TODO:

# Quiz 18: 打印 sst_grid 数组的维度(ndim)、形状(shape)和元素总数(size)。
# TODO:

# Quiz 19: 假设有一个包含 12 个月平均温度的 1D 数组 (长度为12)，请将其 reshape 为 3x4 的数组（代表3个季度，每个季度4个月）。
# temp_12 = np.arange(15, 27)
# TODO:

# Quiz 20: 索引：提取 salinity 数组中的第 3 个元素（注意索引从0开始）。
# TODO:

# Quiz 21: 索引：提取 sst_grid 数组中心位置 (第2行，第2列) 的元素。
# TODO:

# Quiz 22: 切片：提取 salinity 数组的前 3 个元素。
# TODO:

# Quiz 23: 切片：提取 sst_grid 数组的第 1 列所有数据。
# TODO:

# Quiz 24: 标量运算：假设全球变暖导致温度上升 1.5 度，给 sst_grid 中的每个元素都加上 1.5，结果存入 sst_warmed。
# TODO:

# Quiz 25: 数组运算：计算 sst_warmed 与原始 sst_grid 之间的温差（温度异常计算基础）。
# TODO:

## 单元三：其他工具（Matplotlib）基础操作 (Quiz 26 - 35)
# 本单元复习科学可视化的基础，特别是温盐图、剖面图和空间网格的绘制。

# Quiz 26: 导入 matplotlib.pyplot 并简写为 plt。
# TODO:

# 准备绘图数据
# depths_plot = np.array([0, 50, 100, 200, 500, 1000])
# temps_plot = np.array([25.0, 24.5, 22.0, 18.0, 10.0, 4.0])

# Quiz 27: 绘制一个简单的折线图，x 轴为温度，y 轴为深度（模拟海洋温度垂直剖面）。
# TODO:

# Quiz 28: 为上述图表添加标题："Temperature Depth Profile"。
# TODO:

# Quiz 29: 为 x 轴添加标签 "Temperature (C)"，为 y 轴添加标签 "Depth (m)"。
# TODO:

# Quiz 30: 在海洋学中，深度通常向下增加。请将 y 轴反转 (提示: gca().invert_yaxis())，最后调用 plt.show() 显示图像。
# TODO:

# 准备温盐散点数据
# T = np.random.uniform(10, 30, 50)
# S = np.random.uniform(33, 35, 50)

# Quiz 31: 绘制温度(T)和盐度(S)的散点图 (T-S Diagram)，x轴为盐度，y轴为温度。
# TODO:

# Quiz 32: 绘制一组随机生成的海表高度异常(SLA)数据的直方图，设置 bins=20。
# sla_data = np.random.randn(1000) * 0.1
# TODO:

# 准备 2D 网格数据
# sst_field = np.random.rand(20, 20) * 10 + 20

# Quiz 33: 使用 plt.imshow() 将 2D 数组 sst_field 绘制为热力图 (伪彩色图)，设置 colormap 为 'coolwarm'。
# TODO:

# Quiz 34: 为上一步的热力图添加颜色条 (Colorbar)，并标明单位 "SST (°C)"。
# TODO:

# Quiz 35: 将绘制好的热力图保存为当前目录下的高清图片文件 'sst_spatial_map.png'，设置 dpi=300。
# TODO:

## 单元四：NumPy 矢量化进阶操作 (Quiz 36 - 50)
# 抛弃 for 循环，使用 NumPy 提供的矢量化操作来高效处理大规模矩阵。

# 生成测试数据：模拟 100x100 的海表温度场 (含部分极端值和缺失值)
# sst_large = np.random.normal(loc=25, scale=3, size=(100, 100))
# sst_large[10:15, 10:15] = np.nan  # 模拟云遮挡导致的缺失数据

# Quiz 36: 计算 sst_large 数组的全局均值（暂不处理 NaN）。你会发现结果是 nan。
# TODO:

# Quiz 37: 计算 sst_large 中非 NaN 部分的全局均值。(提示：使用 np.nanmean)
# TODO:

# Quiz 38: 找出整个温度场中的最大值和最小值 (同样需要忽略 NaN)。
# TODO:

# Quiz 39: 计算整个温度场的标准差，以评估空间温度的离散程度。
# TODO:

# Quiz 40: 降维聚合：计算每一列的平均值（模拟纬向平均 Zonal Mean），返回一个长度为 100 的 1D 数组。
# TODO:

# Quiz 41: 布尔掩码 (Boolean Masking)：创建一个掩码矩阵 is_heatwave，判断 sst_large 中哪些像素的温度大于 30°C。
# TODO:

# Quiz 42: 使用掩码提取出所有发生热浪 (>30°C) 的具体温度值，结果将是一个 1D 数组。
# TODO:

# Quiz 43: 条件替换 (np.where)：将 sst_large 中所有小于 20°C 的值替换为 0，大于等于 20°C 的保留原值。
# TODO:

# 准备广播机制 (Broadcasting) 数据
# daily_sst = np.random.rand(30, 50, 50) * 5 + 20   # 30天，50x50的空间网格
# clim_mean = np.random.rand(50, 50) + 22           # 气候态平均矩阵，50x50

# Quiz 44: 广播机制：在不使用 for 循环的情况下，计算每一天、每个格点的温度距平 (Anomaly)，即从 daily_sst 中减去 clim_mean。
# TODO:

# Quiz 45: 统计运算：计算上一步算出的三维距平矩阵中，有多少个网格点的距平值大于 2.0 °C。(提示：(条件).sum())
# TODO:

# Quiz 46: 梯度计算：使用 np.gradient 计算 2D 温度场 clim_mean 在 x 和 y 方向的梯度（模拟寻找海洋锋面）。
# TODO:

# Quiz 47: 缺失值识别：找出一个布尔数组，标明 sst_large 中哪些位置是 NaN (缺失观测)。
# TODO:

# Quiz 48: 缺失值填充：将 sst_large 中的所有 NaN 替换为该矩阵的有效全局均值 (第37题的结果)。(提示：可以将掩码和索引结合使用，或者使用 np.nan_to_num)
# TODO:

# Quiz 49: 矩阵乘法：提取 clim_mean 的前 3x3 子矩阵 A，与自身转置 A.T 进行点乘操作 (Dot Product)。这是神经网络中线性层的基础运算。
# TODO:

# Quiz 50: 高级索引 (Fancy Indexing)：假设你有几个特定浮标的坐标索引 lats_idx = [5, 12, 45]，lons_idx = [10, 20, 30]，请从 clim_mean 中一次性提取这三个离散位置的温度值。
# TODO:

# 恭喜完成！`;

const linearModelL31Template = `# L3.1 线性模型及其优化（Q1 + Q2）
# 说明：本练习使用合成数据，聚焦线性模型、格点搜索与梯度下降。
# 请在带有 # TODO: 的注释下方补全代码。

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

## 单元一：数据准备与可视化
# 真实参数（用于生成训练样本）
theta0_true = 0.20
theta1_true = 0.70
n_samples = 120

# Quiz 1: 生成输入 x（范围建议在 [-1.2, 1.2]）与高斯噪声 noise
# TODO:

# Quiz 2: 构造观测 y = theta1_true * x + theta0_true + noise
# TODO:

# Quiz 3: 绘制散点图 (x, y)，并标注标题 "Synthetic dSWH vs dWS"
# TODO:

## 单元二：Q1 一参数模型 + 格点搜索
# 模型：y_hat = theta1 * x（此处不考虑偏置）
theta1_candidates = np.arange(-1.5, 1.5, 0.01)

# Quiz 4: 用 for 循环计算每个候选 theta1 的代价 J(theta1)
# 代价函数：J = mean((y_hat - y)^2) / 2
# TODO:

# Quiz 5: 通过 np.argmin 找到最优 theta1_loop 与最小代价 jmin_loop
# TODO:

# Quiz 6: 使用向量化方式再次计算所有候选参数的代价（不要使用 for）
# 提示：可使用 theta1_candidates[np.newaxis, :] 与 x[:, np.newaxis]
# TODO:

# Quiz 7: 对比循环版和向量化版结果（最优 theta1 和最小 J）
# TODO:

# Quiz 8: 可视化 J-theta1 曲线，并标记最优点
# TODO:

## 单元三：Q2 二参数模型 + 梯度下降
# 模型：y_hat = theta1 * x + theta0
theta0 = 0.0
theta1 = 0.0
alpha = 0.08
n_iters = 200

theta0_hist = []
theta1_hist = []
J_hist = []

# Quiz 9: 在迭代中补全以下步骤
# 1) 前向计算 y_hat
# 2) 计算代价 J
# 3) 计算梯度 dtheta0, dtheta1
# 4) 更新参数
# 5) 记录历史
# TODO:

# Quiz 10: 绘制损失收敛曲线（横轴 iter，纵轴 J）
# TODO:

# Quiz 11: 绘制拟合结果（散点 + 训练后拟合直线）
# TODO:

## 单元四：结果对照与误差分析
# Quiz 12: 使用 np.polyfit(x, y, 1) 得到对照参数 theta1_ref, theta0_ref
# TODO:

# Quiz 13: 打印梯度下降参数与 polyfit 参数，并比较差异
# TODO:

# Quiz 14: 计算两种方法的 MSE 并输出
# TODO:

# 恭喜完成 L3.1！`;

const manualNnL41Template = `# L4.1 手搓 NN（主线：3 节点 + 挑战：10 节点）
# 浏览器可运行版本：仅依赖 NumPy / Matplotlib（不依赖 netCDF4 与外部 Data 文件）
# 请在 # TODO: 处补全代码

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

## 单元一：激活函数与合成数据（Q1 / Q2）
# Quiz 1: 实现 Sigmoid 激活函数
def sigmoid(x):
    # TODO:
    pass

# Quiz 2: 实现 ReLU 激活函数
def relu(x):
    # TODO:
    pass

# 合成数据：y = 0.55 * x + 0.22 + noise
n_samples = 180
x_all = np.linspace(-1.2, 1.2, n_samples)
noise = np.random.normal(0, 0.06, size=n_samples)
y_all = 0.55 * x_all + 0.22 + noise

# mapminmax 到 [0, 1]，便于 sigmoid 输出拟合
y_min, y_max = y_all.min(), y_all.max()
y_all = (y_all - y_min) / (y_max - y_min)

# 划分训练 / 测试
X, Y = x_all[40:], y_all[40:]
X_test, Y_test = x_all[:40], y_all[:40]

## 单元二：3 节点网络前向传播（Q3）
# 网络结构：输入 1 -> 隐藏层 3 -> 输出 1
w1 = np.random.randn(3) * 0.5
b1 = np.random.randn(3) * 0.1
w2 = np.random.randn(3) * 0.5
b2 = np.random.randn() * 0.1

# Quiz 3: 实现前向传播
# 提示：z1 = w1 * x + b1，g1 = sigmoid(z1)，z2 = dot(w2, g1) + b2，yhat = sigmoid(z2)
def forward_propagation(x, w1, b1, w2, b2):
    # TODO:
    pass

## 单元三：逐样本反向传播与参数更新（Q4）
alpha = 0.12
epochs = 600
loss_hist = []

# Quiz 4: 完成训练循环
# 1) 逐样本前向传播
# 2) 计算 MSE 并记录每轮平均损失
# 3) 反向传播并更新 w1 / b1 / w2 / b2
# 4) 每隔若干 epoch 打印损失
for epoch in range(epochs):
    # TODO:
    pass

## 单元四：训练曲线与拟合效果
# Quiz 5: 绘制训练损失曲线（建议 semilogy）
# TODO:

# Quiz 6: 绘制训练集/测试集散点与模型拟合曲线
# TODO:

## 单元五：10 节点挑战题（可选）
# Challenge: 将隐藏层从 3 节点改为 10 节点
# 1) 调整参数维度
# 2) 复用前向/反向流程
# 3) 对比收敛速度与拟合表现
# TODO:

# 恭喜完成 L4.1！`;

const vectorizedNnL42Template = `# L4.2 手搓 NN 进阶（向量化版）
# 目标：在 L4.1 的基础上，把“逐样本循环”升级为“批量向量化计算”

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

## 单元一：先看 L4.1 -> L4.2 的迁移差异
# 差异 1：输入从单个 x 变为批量 X_batch
# 差异 2：前向传播输出从标量变为向量/矩阵
# 差异 3：梯度从逐样本累加改为一次 batch 计算
# 差异 4：训练速度更快，代码更贴近深度学习框架

# 保持与 L4.1 一致的数据
n_samples = 180
x_all = np.linspace(-1.2, 1.2, n_samples)
noise = np.random.normal(0, 0.06, size=n_samples)
y_all = 0.55 * x_all + 0.22 + noise

y_min, y_max = y_all.min(), y_all.max()
y_all = (y_all - y_min) / (y_max - y_min)

X, Y = x_all[40:], y_all[40:]
X_test, Y_test = x_all[:40], y_all[:40]

# 初始化参数（隐藏层 3 节点）
w1 = np.random.randn(3) * 0.5
b1 = np.random.randn(3) * 0.1
w2 = np.random.randn(3) * 0.5
b2 = np.random.randn() * 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## 单元二：批量前向传播与批量梯度（完整模板）
# Quiz 1: 实现批量前向传播
# X_batch shape: (N,)
# z1 shape: (N, 3)
# g1 shape: (N, 3)
# z2 shape: (N,)
# yhat shape: (N,)
def forward_batch(X_batch, w1, b1, w2, b2):
    # TODO:
    pass

# Quiz 2: 实现批量反向传播
# 目标：J = mean((yhat - y_true)^2)
# 可用链式法则：
# dJ/dyhat = 2*(yhat-y)/N
# dyhat/dz2 = yhat*(1-yhat)
# z2 = g1 @ w2 + b2
# z1 = X[:,None]*w1[None,:] + b1[None,:]
def backward_batch(X_batch, y_true, yhat, z1, g1, w2):
    # TODO:
    # return dw1, db1, dw2, db2
    pass

## 单元三：向量化训练、收敛曲线与效果对照
alpha = 0.12
epochs = 600
loss_hist = []

# Quiz 3: 完成向量化训练循环，并记录每个 epoch 的损失
for epoch in range(epochs):
    # TODO:
    pass

# Quiz 4: 绘制损失曲线，并与 L4.1 结果做定性对照
# TODO:

# Quiz 5: 可视化训练集/测试集 + 模型预测曲线
# TODO:

# Quiz 6: 输出测试集 MSE，并和 L4.1 对比
# TODO:

# 恭喜完成 L4.2！`;

export const labExercises: LabExercise[] = [
  // 基础练习（对应 Student_Notebook.ipynb 课前热身 50 Quizzes）
  {
    id: 'L1',
    title: '课前热身编程练习 (50 Quizzes)',
    description: '复习 Python、NumPy、Matplotlib 核心操作，掌握向量化编程思维，为后续深度学习处理海洋与气候数据打基础。',
    difficulty: 'beginner',
    category: '基础练习',
    topics: ['Python 基本操作', 'NumPy 基础与进阶', 'Matplotlib 可视化', '向量化编程'],
    codeTemplate: notebookBasicTemplate,
    hints: [
      '在带有 # TODO: 的注释下方编写代码',
      '单元一：列表、字典、循环、函数、列表推导、f-string',
      '单元二：np.array、arange、linspace、索引切片、reshape、标量/数组运算',
      '单元三：plt 折线图、散点图、直方图、imshow、colorbar、invert_yaxis、savefig',
      '单元四：nanmean、布尔掩码、np.where、广播、np.gradient、np.nan_to_num、矩阵乘法、高级索引'
    ]
  },
  {
    id: 'L2.1',
    title: 'NumPy地转差分',
    description: '（内容暂不开放）',
    difficulty: 'intermediate',
    category: '海洋数据处理',
    topics: ['地转流', '差分', '海洋物理'],
  },
  // 神经网络基础
  {
    id: 'L3.1',
    title: '线性模型及其优化',
    description: '基于合成海洋数据完成 Q1 格点搜索与 Q2 梯度下降，掌握代价函数最小化与参数优化流程。',
    difficulty: 'intermediate',
    category: '神经网络基础',
    topics: ['线性回归', '梯度下降', '代价函数'],
    codeTemplate: linearModelL31Template,
    hints: [
      '先执行单元一，确保 x 与 y 已生成，再进入 Q1/Q2 优化步骤',
      'Q1 代价函数建议统一写成 J = np.mean((y_hat - y) ** 2) / 2，便于循环版和向量化版对照',
      '向量化计算时，注意维度扩展：x[:, np.newaxis] 与 theta_candidates[np.newaxis, :]',
      'Q2 梯度下降常用梯度：dtheta1 = mean((y_hat - y) * x)，dtheta0 = mean(y_hat - y)',
      '学习率 alpha 过大会震荡，过小会收敛慢，可在 0.02~0.1 区间尝试',
      '结果对照可用 np.polyfit(x, y, 1)；返回顺序是 slope(theta1), intercept(theta0)',
      '建议同时观察损失曲线与拟合直线，确认“数值收敛”和“拟合效果”一致'
    ],
  },
  {
    id: 'L4.1',
    title: '手搓NN基础练习',
    description: '完成 3 节点手搓神经网络的前向与反向传播，并在挑战题中扩展到 10 节点结构。',
    difficulty: 'advanced',
    category: '神经网络基础',
    topics: ['反向传播', '梯度下降', '非线性拟合'],
    codeTemplate: manualNnL41Template,
    hints: [
      '先把 Sigmoid/ReLU 单独测通，再接入前向传播，定位问题会更快。',
      'L4.1 主线固定为 3 节点隐藏层，确保你先跑通主线再做挑战题。',
      '若输出层使用 sigmoid，建议将 y 做 mapminmax 到 [0,1] 再训练。',
      '反向传播可先按“输出层 -> 隐藏层”顺序手推公式，再写代码。',
      '每隔若干 epoch 打印一次 MSE，并用 semilogy 看收敛趋势。',
      '10 节点挑战建议只改参数维度，不重写训练主逻辑。',
      '对比 3 节点与 10 节点时，重点观察欠拟合/过拟合与收敛速度。'
    ],
  },
  {
    id: 'L4.2',
    title: '手搓NN进阶 - 向量化',
    description: '在 L4.1 基础上完成批量向量化前向与反向传播，实现更高效的训练流程。',
    difficulty: 'advanced',
    category: '神经网络基础',
    topics: ['向量化', '批量梯度下降', '性能优化'],
    codeTemplate: vectorizedNnL42Template,
    hints: [
      '先明确每个张量的 shape，再下手写向量化公式，能避免大部分 bug。',
      '隐藏层可用 `X[:, None]` 与参数广播，得到 z1 的 `(N, hidden)` 结构。',
      '向量化本质是把逐样本梯度求和，改成一次 Batch GD 计算。',
      '建议先写 forward_batch 并验证输出，再实现 backward_batch。',
      '与 L4.1 对照时，保持相同 epoch 与学习率，便于公平比较。',
      '若训练不稳定，优先检查 alpha 和梯度维度是否匹配。'
    ],
  },
  // PyTorch实践
  {
    id: 'L4.1-PyTorch',
    title: 'NN模型用PyTorch',
    description: '（内容暂不开放）',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['nn.Module', '优化器', '训练循环'],
    hidden: true
  },
  {
    id: 'L5.1',
    title: 'PyTorch基础练习',
    description: '（内容暂不开放）',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['DataLoader', 'Dataset', '批量训练'],
    hidden: true
  },
  {
    id: 'L5.2',
    title: 'PyTorch进阶练习',
    description: '（内容暂不开放）',
    difficulty: 'intermediate',
    category: 'PyTorch实践',
    topics: ['模型保存', 'GPU训练', '学习率调度'],
    hidden: true
  },
  // 深度学习应用
  {
    id: 'LeNet',
    title: 'LeNet手写字体识别',
    description: '（内容暂不开放）',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['CNN', 'LeNet', 'MNIST', '分类'],
    hidden: true
  },
  // 新增：U-Net练习
  {
    id: 'L7.1',
    title: 'U-Net海洋数据重建',
    description: '（内容暂不开放）',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['U-Net', 'OHC重建', '深海遥感'],
    hidden: true
  },
  // 新增：LSTM练习
  {
    id: 'L8.1',
    title: 'LSTM径流预测',
    description: '（内容暂不开放）',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['LSTM', '时间序列', '预测'],
    hidden: true
  },
  // 新增：PINN练习
  {
    id: 'L11.1',
    title: 'PINN求解热传导方程',
    description: '（内容暂不开放）',
    difficulty: 'advanced',
    category: '深度学习应用',
    topics: ['PINN', 'PDE', '物理约束'],
    hidden: true
  }
];

