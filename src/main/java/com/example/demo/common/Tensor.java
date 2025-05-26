package com.example.demo.common;

import java.util.Arrays;

// 张量类，用于表示多维数组
public class Tensor {
    private double[][][][] data; // 4D 张量数据，维度为 [batch, channel, height, width]
    private double[][][][] grad;  // 梯度数据

    // 构造函数，初始化张量数据
    public Tensor(int batch, int channel, int height, int width) {
        this.data = new double[batch][channel][height][width];
        this.grad = new double[batch][channel][height][width];
    }

    // 获取批次大小
    public int getBatchSize() {
        return data.length;
    }

    // 获取高度
    public int getHeight() {
        return data[0][0].length;
    }

    // 获取宽度
    public int getWidth() {
        return data[0][0][0].length;
    }

    // 获取深度（通道数）
    public int getDepth() {
        return data[0].length;
    }

    // 将张量数据重塑为矩阵
    public double[][] reshapeToMatrix() {
        int batch = getBatchSize();
        int channel = getDepth();
        int height = getHeight();
        int width = getWidth();
        int rows = batch * channel;
        int cols = height * width;
        double[][] matrix = new double[rows][cols];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                int rowIndex = b * channel + c;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int colIndex = h * width + w;
                        matrix[rowIndex][colIndex] = data[b][c][h][w];
                    }
                }
            }
        }
        return matrix;
    }

    // 为张量添加偏置
    public void addBias(Tensor bias) {
        int batch = getBatchSize();
        int channel = getDepth();
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        data[b][c][h][w] += bias.data[0][c][0][0];
                    }
                }
            }
        }
    }

    // 逐元素映射操作
    public Tensor map(java.util.function.DoubleUnaryOperator lambda) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = lambda.applyAsDouble(data[b][c][h][w]);
                    }
                }
            }
        }
        return result;
    }

    // 张量相加
    public Tensor add(Tensor other) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] + other.data[b][c][h][w];
                    }
                }
            }
        }
        return result;
    }

    // 张量相加（标量加法）
    public Tensor add(double scalar) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] + scalar;
                    }
                }
            }
        }
        return result;
    }

    // 张量相减
    public Tensor subtract(Tensor other) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] - other.data[b][c][h][w];
                    }
                }
            }
        }
        return result;
    }

    // 张量乘法（张量乘以张量）
    public Tensor multiply(Tensor other) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] * other.data[b][c][h][w];
                    }
                }
            }
        }
        return result;
    }

    // 张量乘法（张量乘以标量）
    public Tensor multiply(double value) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] * value;
                    }
                }
            }
        }
        return result;
    }

    // 张量除法（除以标量）
    public Tensor divide(double value) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] / value;
                    }
                }
            }
        }
        return result;
    }

    // 张量除法（除以 Tensor）
    public Tensor divide(Tensor other) {
        Tensor result = new Tensor(getBatchSize(), getDepth(), getHeight(), getWidth());
        for (int b = 0; b < getBatchSize(); b++) {
            for (int c = 0; c < getDepth(); c++) {
                for (int h = 0; h < getHeight(); h++) {
                    for (int w = 0; w < getWidth(); w++) {
                        result.data[b][c][h][w] = data[b][c][h][w] / other.data[b][c][h][w];
                    }
                }
            }
        }
        return result;
    }

    // 张量平方
    public Tensor square() {
        return map(x -> x * x);
    }

    // 张量开方
    public Tensor sqrt() {
        return map(Math::sqrt);
    }

    // 张量求和
    public double sum() {
        double sum = 0;
        for (double[][][] batch : data) {
            for (double[][] channel : batch) {
                for (double[] row : channel) {
                    for (double val : row) {
                        sum += val;
                    }
                }
            }
        }
        return sum;
    }

    // 设置张量数据
    public void setData(double[][][][] data) {
        this.data = data;
    }

    // 获取张量数据
    public double[][][][] getData() {
        return data;
    }

    // 设置单个元素的值
    public void setElement(int batchIndex, int channelIndex, int heightIndex, int widthIndex, double value) {
        data[batchIndex][channelIndex][heightIndex][widthIndex] = value;
    }

    // 获取单个元素的值
    public double getElement(int batchIndex, int channelIndex, int heightIndex, int widthIndex) {
        return data[batchIndex][channelIndex][heightIndex][widthIndex];
    }

    @Override
    public String toString() {
        return "Tensor{" +
                "data=" + Arrays.deepToString(data) +
                '}';
    }
}    