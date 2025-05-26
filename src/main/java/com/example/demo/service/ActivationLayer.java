package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Activation;
import com.example.demo.face.Layer;

import java.util.HashMap;
import java.util.Map;

// 激活层实现
public class ActivationLayer implements Layer {
    private Activation activation;

    // 构造函数，初始化激活层的激活函数
    public ActivationLayer(Activation activation) {
        this.activation = activation;
    }

    @Override
    public Tensor forward(Tensor input, boolean isTraining) {
        // 应用激活函数
        return activation.apply(input);
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        // 计算激活函数的导数并与输出梯度相乘
        return outputGradient.multiply(activation.derivative(outputGradient));
    }

    @Override
    public Map<String, Tensor> getParameters() {
        // 激活层没有可训练参数
        return new HashMap<>();
    }

    @Override
    public Map<String, Tensor> getGradients() {
        // 激活层没有梯度
        return new HashMap<>();
    }
}