package com.example.demo.face;

import com.example.demo.face.Layer;

import java.util.List;

// 优化器接口
public interface Optimizer {
    // 更新层的参数
    void update(List<Layer> layers);
}