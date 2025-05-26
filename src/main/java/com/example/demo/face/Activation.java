package com.example.demo.face;

import com.example.demo.common.Tensor;

// 激活函数接口
public interface Activation {
    // 应用激活函数
    Tensor apply(Tensor input);
    // 计算激活函数的导数
    Tensor derivative(Tensor input);
}