package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;


public class Main {
    public static void main(String[] args) {
        // 初始化超参数
        double learningRate = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        int epochs = 5;
        int batchSize = 16;
        int inputChannels = 1;
        int inputHeight = 28;
        int inputWidth = 28;
        int numClasses = 10;

        // 创建模型
        CNNModel model = new CNNModel();

        // 添加层
        Layer simpleLayer = new SimpleLayer();
        model.add(simpleLayer);

        // 编译模型
        AdamOptimizer optimizer = new AdamOptimizer(learningRate, beta1, beta2, epsilon);
        CrossEntropyLoss lossFunction = new CrossEntropyLoss();
        model.compile(lossFunction, optimizer);

        // 模拟训练数据
        Tensor input = new Tensor(batchSize, inputChannels, inputHeight, inputWidth);
        Tensor target = new Tensor(batchSize, numClasses, 1, 1);

        // 初始化输入数据
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int h = 0; h < inputHeight; h++) {
                    for (int w = 0; w < inputWidth; w++) {
                        input.setElement(b, c, h, w, Math.random());
                    }
                }
            }
        }

        // 初始化目标数据（简单随机设置类别）
        for (int b = 0; b < batchSize; b++) {
            int randomClass = (int) (Math.random() * numClasses);
            for (int c = 0; c < numClasses; c++) {
                target.setElement(b, c, 0, 0, c == randomClass ? 1 : 0);
            }
        }

        // 训练模型
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.trainStep(input, target);
            double loss = lossFunction.compute(model.forward(input, false), target);
            System.out.printf("Epoch %d, Loss: %.6f%n", epoch + 1, loss);
        }
    }
}