# ResNet32模型
---

## 🌐模型介绍:
>这是一个用PyTorch实现的ResNet32，一个用于图像分类的深度卷积神经网络。  
>ResNet是由He等人在深度残差学习的图像识别中提出的，它引入了残差连接来解决非常深的网络的退化问题。  
>ResNet在各种图像识别任务和基准上都取得了最先进的结果。  

## 🌐库要求:
>Python 3.6或更高版本  
>PyTorch 1.8或更高版本  
>Torchvision 0.9或更高版本  
>Numpy 1.19或更高版本  

## 🌐更多提升：
>该代码只支持CIFAR-10数据集，它相对较小而简单。可以测试模型在其他数据集上的表现，例如ImageNet或COCO。  
>  
>该代码只实现了ResNet32，它是ResNet的一个相对浅层的变体。实现其他变体的ResNet，   
>例如ResNet50或ResNet101，并比较它们的性能和效率。  
>  
>该代码只使用了基本的数据增强技术，例如随机裁剪和翻转。使用更高级的数据增强技术，    
>例如Cutout或Mixup，来进一步提高模型的鲁棒性和泛化能力。  

---

# ResNet32模型代码部分以及局部注释
---

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
  
  
    # 把残差连接补充到 Block 的 forward 函数中
    class Block(nn.Module):
        def __init__(self, dim, out_dim, stride) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_dim)
            self.relu2 = nn.ReLU()
  
        def forward(self, x):
            x = self.conv1(x) # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，改变通道数和分辨率
            x = self.bn1(x) # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
            x = self.relu1(x) # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
            x = self.conv2(x) # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，保持通道数和分辨率
            x = self.bn2(x) # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
            x = self.relu2(x) # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
            return x
  
  
    class ResNet32(nn.Module):
         def __init__(self, in_channel=64, num_classes=2):
            super().__init__()
            self.num_classes = num_classes
            self.in_channel = in_channel
    
            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
            self.maxpooling = nn.MaxPool2d(kernel_size=2)
            self.last_channel = in_channel
  
            self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
            self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
            self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
            self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)
  
            self.avgpooling = nn.AvgPool2d(kernel_size=2)
            self.classifier = nn.Linear(4608, self.num_classes)
  
        def _make_layer(self, in_channel, num_blocks, stride):
            layer_list = [Block(self.last_channel, in_channel, stride)]
            self.last_channel = in_channel
            for i in range(1, num_blocks):
                b = Block(in_channel, in_channel, stride=1)
                layer_list.append(b)
            return nn.Sequential(*layer_list)
  
        def forward(self, x):
            x = self.conv1(x)  # [bs , 64 , 56 , 56 ] 特征提取过程
            x = self.maxpooling(x)  # [bs , 64 , 28 , 28 ] 池化，降低分辨率和计算量
            x = self.layer1(x) # [bs , 64 , 28 , 28 ] 残差层，提取特征，保持通道数和分辨率
            x = self.layer2(x) # [bs , 128 , 14 , 14 ] 残差层，提取特征，改变通道数和分辨率
            x = self.layer3(x) # [bs , 256 , 7 , 7 ] 残差层，提取特征，改变通道数和分辨率
            x = self.layer4(x) # [bs , 512 , 4 , 4 ] 残差层，提取特征，改变通道数和分辨率
            x = self.avgpooling(x) # [bs , 512 , 2 , 2 ] 平均池化，降低分辨率和计算量
            x = x.view(x.shape[0], -1) # [bs , 2048 ] 展平张量，准备分类
            x = self.classifier(x) # [bs , num_classes ] 全连接层，输出类别概率
            output = F.softmax(x) # [bs , num_classes ] softmax函数，归一化概率
  
            return output
  
  
    if __name__=='__main__':
        t = torch.randn([8 , 3 , 224 , 224 ]) # 随机生成一个8个样本的输入张量
        model = ResNet32() # 创建一个ResNet32模型实例
        out = model(t) # 调用模型的forward方法得到输出张量
        print(out.shape) # 打印输出张量的形状：[8 , num_classes ]

---

📃其中不同的参数具有不同的作用与意义：  
  
    ⒈out_dim：会影响模型的表达能力，通常越大越好，但也会增加模型的参数量和计算量。  

    ⒉h/stride和w/stride：会影响模型的感受野，通常越小越好，但也会增加模型的深度和复杂度。  

    ⒊kernel_size：会影响模型的局部特征提取能力，通常越大越好，但也会增加模型的参数量和计算量。  

    ⒋stride：会影响模型的分辨率和感受野，通常越大越好，但也会降低模型的特征保留能力。  

    ⒌padding：会影响模型的边缘特征提取能力，通常需要根据kernel_size和stride进行合理的设置。  

**这些参数之间存在一定的平衡和折中，需要根据不同的任务和数据进行调整。**

---

📜Pytorch搭建模型模板的语法问题：

  搭建模型的模版时，给出代码如下：  

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
  
    class xxxNet(nn.Module):
    	def __init__(self):
	         pass
    	def forward(x):
	          return x
            
  在第6行处，需要在__init__方法中调用父类的__init__方法，例如super(xxxNet, self).__init__()否则模型就无法继承父类的属性和方法，  
  例如parameters()，to()等，这会影响到模型的初始化和训练。  
  
  在第8行处，需要在forward方法的参数列表中加上self，例如def forward(self, x):否则forward方法就无法访问模型的实例属性和方法，  
  例如self.xxx，这会导致模型无法正常运行。  
  
  所以修改后的模板应该如下：  
  
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
  
    class xxxNet(nn.Module):
    	def __init__(self):
    	      super(xxxNet, self).__init__() # 调用父类的__init__方法
    	def forward(self, x): # 加上self参数
	          return x
            
---

# 补充（仍在补充）：
---

## ✒残差连接  
 
残差连接是一种跳跃连接的类型，它可以让神经网络学习输入层的残差函数，而不是学习无参考的函数。  
残差连接的直观想法是，相比于让一堆非线性层去拟合一个复杂的函数，让它们去拟合一个残差函数可能更容易。  
残差连接可以有效地解决深度神经网络训练过程中的梯度消失和梯度爆炸等问题。  
残差连接在不同的模型中都有广泛的应用，例如计算机视觉中的ResNet，自然语言处理中的Transformer，强化学习中的AlphaZero和蛋白质结构预测中的AlphaFold等。  
  
所以它是一种简单而有效的技术，可以提高神经网络的性能和泛化能力。它也可以帮助我们设计更深更复杂的模型，来解决更难的问题。  

## ✒ResNet32模型特点

它定义了一个ResNet32类，继承了nn.Module类。  
它在初始化函数中设置了四个卷积块，每个卷积块包含若干个残差单元。第二、三、四个卷积块中分别有2、3、5个残差单元，这与ResNet32的结构相符。  
它在前向传播函数中依次调用了卷积层、池化层、四个卷积块、平均池化层和分类器，最后返回输出。  
它在主函数中创建了一个ResNet32模型的实例，并用随机生成的张量作为输入，测试了模型的输出形状。  

## ✒鲁棒性与泛化能力

**鲁棒性**  是指一个系统或模型对于输入的扰动或变化的抵抗能力，也就是说，当输入有一些噪声或异常时，系统或模型的输出能否保持稳定和正确。  
**泛化能力**  是指一个系统或模型对于未知数据的预测能力，也就是说，当输入是训练集以外的数据时，系统或模型的输出能否保持准确和有效。  
  
一般来说，鲁棒性和泛化能力都是用来衡量系统或模型的性能和质量的指标，它们之间有一定的联系，但也有一定的区别。例如：  

>一个系统或模型可能具有很强的鲁棒性，但泛化能力很差，这意味着它对于训练集中的数据很稳健，但对于训练集以外的数据很不适应。  
>一个系统或模型可能具有很强的泛化能力，但鲁棒性很差，这意味着它对于训练集以外的数据很适应，但对于训练集中的数据很敏感。  
>一个系统或模型可能具有适中的鲁棒性和泛化能力，这意味着它对于各种数据都有一定的适应性和稳健性。  
>一个系统或模型可能具有很强的鲁棒性和泛化能力，这意味着它对于各种数据都有很高的适应性和稳健性。  
  
因此，在设计和评估一个系统或模型时，需要综合考虑它的鲁棒性和泛化能力，并根据不同的场景和需求进行权衡和优化。
