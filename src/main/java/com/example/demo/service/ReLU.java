package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Activation;

// ReLU 激活函数实现
public class ReLU implements Activation {
    @Override
    public Tensor apply(Tensor input) {
        // 逐元素计算 max(0, x)
        return input.map(x -> x > 0 ? x : 0);
    }

    @Override
    public Tensor derivative(Tensor input) {
        // 梯度为 1（x > 0）或 0（x ≤ 0）
        return input.map(x -> x > 0 ? 1 : 0);
    }
}