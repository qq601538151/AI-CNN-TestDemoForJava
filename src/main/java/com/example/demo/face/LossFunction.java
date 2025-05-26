package com.example.demo.face;

import com.example.demo.common.Tensor;

// 损失函数接口
public interface LossFunction {
    // 计算损失
    double calculate(Tensor output, Tensor target);
    // 计算损失的梯度
    Tensor gradient(Tensor output, Tensor target);
}