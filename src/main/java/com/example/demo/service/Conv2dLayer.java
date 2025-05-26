package com.example.demo.service;

import com.example.demo.common.Matrix;
import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;

import java.util.HashMap;
import java.util.Map;

// 卷积层实现
public class Conv2dLayer implements Layer {
    private int kernelSize, stride, padding;
    private Tensor weights, bias;
    private Tensor inputCol, output;
    private Tensor input;

    // 构造函数，初始化卷积层参数
    public Conv2dLayer(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        // 初始化权重张量
        this.weights = new Tensor(outChannels, inChannels, kernelSize, kernelSize);
        // 初始化偏置张量
        this.bias = new Tensor(outChannels, 1, 1, 1);
    }

    @Override
    public Tensor forward(Tensor input, boolean isTraining) {
        this.input = input;
        int n = input.getBatchSize();
        int h = input.getHeight();
        int w = input.getWidth();
        // 将输入转换为列矩阵
        inputCol = im2col(input.getData(), h, w, kernelSize, stride, padding);
        // 将权重转换为矩阵
        double[][] weightMat = weights.reshapeToMatrix();
        // 将输入列矩阵转换为矩阵
        double[][] inputMat = inputCol.reshapeToMatrix();
        // 矩阵乘法
        double[][] outputMat = Matrix.multiply(weightMat, inputMat);
        int outH = (h + 2 * padding - kernelSize) / stride + 1;
        int outW = (w + 2 * padding - kernelSize) / stride + 1;
        // 将输出矩阵转换为 4D 张量
        output = new Tensor(n, weights.getDepth(), outH, outW);
        output.setData(Matrix.reshapeTo4D(outputMat, n, weights.getDepth(), outH, outW));
        // 添加偏置
        output.addBias(bias);
        return output;
    }

    private Tensor im2col(double[][][][] input, int h, int w, int k, int s, int p) {
        int outH = (h + 2 * p - k) / s + 1;
        int outW = (w + 2 * p - k) / s + 1;
        int batch = input.length;
        int channel = input[0].length;
        // 修正 Tensor 构造函数调用
        Tensor col = new Tensor(batch, channel * k * k, 1, outH * outW);
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                for (int i = 0; i < outH; i++) {
                    for (int j = 0; j < outW; j++) {
                        int colIndex = i * outW + j;
                        for (int ki = 0; ki < k; ki++) {
                            for (int kj = 0; kj < k; kj++) {
                                int inputH = i * s + ki - p;
                                int inputW = j * s + kj - p;
                                double val = 0;
                                if (inputH >= 0 && inputH < h && inputW >= 0 && inputW < w) {
                                    val = input[b][c][inputH][inputW];
                                }
                                int rowIndex = c * k * k + ki * k + kj;
                                // 使用公共方法设置元素值
                                col.setElement(b, rowIndex, 0, colIndex, val);
                            }
                        }
                    }
                }
            }
        }
        return col;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        int n = outputGradient.getBatchSize();
        int h = input.getHeight();
        int w = input.getWidth();
        int outH = outputGradient.getHeight();
        int outW = outputGradient.getWidth();
        double[][] gradMat = outputGradient.reshapeToMatrix();
        double[][] weightMat = weights.reshapeToMatrix();
        // 计算权重梯度
        double[][] weightGradMat = Matrix.multiply(gradMat, Matrix.transpose(inputCol.reshapeToMatrix()));
        Tensor weightGrad = new Tensor(weights.getBatchSize(), weights.getDepth(), weights.getHeight(), weights.getWidth());
        weightGrad.setData(Matrix.reshapeTo4D(weightGradMat, weights.getBatchSize(), weights.getDepth(), weights.getHeight(), weights.getWidth()));
        // 计算输入梯度
        double[][] inputGradMat = Matrix.multiply(Matrix.transpose(weightMat), gradMat);
        Tensor inputGrad = new Tensor(n, input.getDepth(), h, w);
        inputGrad.setData(col2im(inputGradMat, n, input.getDepth(), h, w, kernelSize, stride, padding));
        // 计算偏置梯度
        Tensor biasGrad = new Tensor(bias.getBatchSize(), bias.getDepth(), bias.getHeight(), bias.getWidth());
        for (int b = 0; b < n; b++) {
            for (int c = 0; c < bias.getDepth(); c++) {
                for (int i = 0; i < outH; i++) {
                    for (int j = 0; j < outW; j++) {
                        double currentBiasGrad = biasGrad.getElement(0, c, 0, 0);
                        double outputGradValue = outputGradient.getElement(b, c, i, j);
                        biasGrad.setElement(0, c, 0, 0, currentBiasGrad + outputGradValue);
                    }
                }
            }
        }
        // 更新梯度
        Map<String, Tensor> grads = getGradients();
        grads.put("weights", weightGrad);
        grads.put("bias", biasGrad);
        return inputGrad;
    }

    private double[][][][] col2im(double[][] colMat, int batch, int channel, int h, int w, int k, int s, int p) {
        int outH = (h + 2 * p - k) / s + 1;
        int outW = (w + 2 * p - k) / s + 1;
        double[][][][] im = new double[batch][channel][h][w];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                for (int i = 0; i < outH; i++) {
                    for (int j = 0; j < outW; j++) {
                        int colIndex = i * outW + j;
                        for (int ki = 0; ki < k; ki++) {
                            for (int kj = 0; kj < k; kj++) {
                                int inputH = i * s + ki - p;
                                int inputW = j * s + kj - p;
                                if (inputH >= 0 && inputH < h && inputW >= 0 && inputW < w) {
                                    int rowIndex = c * k * k + ki * k + kj;
                                    im[b][c][inputH][inputW] += colMat[b * channel * k * k + rowIndex][colIndex];
                                }
                            }
                        }
                    }
                }
            }
        }
        return im;
    }

    @Override
    public Map<String, Tensor> getParameters() {
        Map<String, Tensor> params = new HashMap<>();
        params.put("weights", weights);
        params.put("bias", bias);
        return params;
    }

    @Override
    public Map<String, Tensor> getGradients() {
        Map<String, Tensor> grads = new HashMap<>();
        grads.put("weights", new Tensor(weights.getBatchSize(), weights.getDepth(), weights.getHeight(), weights.getWidth()));
        grads.put("bias", new Tensor(bias.getBatchSize(), bias.getDepth(), bias.getHeight(), bias.getWidth()));
        return grads;
    }
}    