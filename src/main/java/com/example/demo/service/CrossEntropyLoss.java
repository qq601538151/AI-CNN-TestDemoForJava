package com.example.demo.service;

import com.example.demo.common.Tensor;
import com.example.demo.face.Loss;

// 交叉熵损失函数
public class CrossEntropyLoss implements Loss {
    @Override
    public double compute(Tensor output, Tensor target) {
        int batchSize = output.getBatchSize();
        int numClasses = output.getDepth();
        double loss = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < numClasses; c++) {
                double y = target.getElement(b, c, 0, 0);
                double yPred = output.getElement(b, c, 0, 0);
                if (y == 1) {
                    loss -= Math.log(yPred + 1e-8);
                }
            }
        }
        return loss / batchSize;
    }

    @Override
    public Tensor gradient(Tensor output, Tensor target) {
        int batchSize = output.getBatchSize();
        int numClasses = output.getDepth();
        Tensor grad = new Tensor(batchSize, numClasses, 1, 1);
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < numClasses; c++) {
                double y = target.getElement(b, c, 0, 0);
                double yPred = output.getElement(b, c, 0, 0);
                grad.setElement(b, c, 0, 0, (yPred - y) / batchSize);
            }
        }
        return grad;
    }
}    