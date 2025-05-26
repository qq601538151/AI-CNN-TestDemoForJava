package com.example.demo.face;

import com.example.demo.common.Tensor;

// 层接口，定义了层的基本操作
public interface Layer {
    // 前向传播方法，接收输入张量和是否为训练模式的标志
    Tensor forward(Tensor input, boolean isTraining);
    // 反向传播方法，接收输出梯度
    Tensor backward(Tensor outputGradient);
    // 获取层的参数，如权重和偏置
    java.util.Map<String, Tensor> getParameters();
    // 获取层的梯度
    java.util.Map<String, Tensor> getGradients();
}