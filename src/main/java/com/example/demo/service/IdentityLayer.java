package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;

import java.util.HashMap;
import java.util.Map;

// 恒等层，用于残差连接
public class IdentityLayer implements Layer {
    @Override
    public Tensor forward(Tensor input, boolean isTraining) {
        return input;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        return outputGradient;
    }

    @Override
    public Map<String, Tensor> getParameters() {
        return new HashMap<>();
    }

    @Override
    public Map<String, Tensor> getGradients() {
        return new HashMap<>();
    }
}