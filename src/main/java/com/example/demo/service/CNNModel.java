package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;
import com.example.demo.face.Loss;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

// CNN 模型类
public class CNNModel {
    private List<Layer> layers;
    private Loss loss;
    private AdamOptimizer optimizer;

    // 构造函数
    public CNNModel() {
        this.layers = new ArrayList<>();
    }

    // 添加层
    public void add(Layer layer) {
        layers.add(layer);
    }

    // 编译模型
    public void compile(Loss loss, AdamOptimizer optimizer) {
        this.loss = loss;
        this.optimizer = optimizer;
    }

    // 获取 layers 属性的公共方法
    public List<Layer> getLayers() {
        return layers;
    }

    // 设置 layers 属性的公共方法
    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    // 前向传播
    public Tensor forward(Tensor input, boolean isTraining) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output, isTraining);
        }
        return output;
    }

    // 反向传播
    public void backward(Tensor outputGradient) {
        Tensor grad = outputGradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }
    }

    // 获取所有层的参数
    public Map<String, Tensor> getParameters() {
        Map<String, Tensor> params = new java.util.HashMap<>();
        for (int i = 0; i < layers.size(); i++) {
            Map<String, Tensor> layerParams = layers.get(i).getParameters();
            for (Map.Entry<String, Tensor> entry : layerParams.entrySet()) {
                params.put("layer" + i + "_" + entry.getKey(), entry.getValue());
            }
        }
        return params;
    }

    // 获取所有层的梯度
    public Map<String, Tensor> getGradients() {
        Map<String, Tensor> grads = new java.util.HashMap<>();
        for (int i = 0; i < layers.size(); i++) {
            Map<String, Tensor> layerGrads = layers.get(i).getGradients();
            for (Map.Entry<String, Tensor> entry : layerGrads.entrySet()) {
                grads.put("layer" + i + "_" + entry.getKey(), entry.getValue());
            }
        }
        return grads;
    }

    // 训练步骤
    public void trainStep(Tensor input, Tensor target) {
        Tensor output = forward(input, true);
        double lossValue = loss.compute(output, target);
        Tensor lossGradient = loss.gradient(output, target);
        backward(lossGradient);
        optimizer.step(getParameters(), getGradients());
    }
}