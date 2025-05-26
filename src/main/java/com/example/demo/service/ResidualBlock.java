package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;

import java.util.HashMap;
import java.util.Map;

// 残差块实现
public class ResidualBlock implements Layer {
    private Layer conv1, conv2, shortcut;

    // 构造函数，初始化残差块的层
    public ResidualBlock(int channels) {
        conv1 = new Conv2dLayer(channels, channels, 3, 1, 1);
        conv2 = new Conv2dLayer(channels, channels, 3, 1, 1);
        shortcut = new IdentityLayer(); // 恒等连接
    }

    @Override
    public Tensor forward(Tensor input, boolean isTraining) {
        Tensor x = conv1.forward(input, isTraining);
        x = new ReLU().apply(x);
        x = conv2.forward(x, isTraining);
        Tensor residual = shortcut.forward(input, isTraining);
        return x.add(residual); // 残差连接
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        // 简单示例，这里只是占位，实际需要实现具体逻辑
        return new Tensor(0, 0, 0, 0);
    }

    @Override
    public Map<String, Tensor> getParameters() {
        Map<String, Tensor> params = new HashMap<>();
        params.putAll(conv1.getParameters());
        params.putAll(conv2.getParameters());
        return params;
    }

    @Override
    public Map<String, Tensor> getGradients() {
        Map<String, Tensor> grads = new HashMap<>();
        grads.putAll(conv1.getGradients());
        grads.putAll(conv2.getGradients());
        return grads;
    }
}