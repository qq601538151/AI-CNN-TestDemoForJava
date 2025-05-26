package com.example.demo.face;


import com.example.demo.common.Tensor;

// 损失函数接口
public interface Loss {
    double compute(Tensor output, Tensor target);
    Tensor gradient(Tensor output, Tensor target);
}    