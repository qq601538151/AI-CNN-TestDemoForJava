package com.example.demo.common;

// 矩阵操作类
public class Matrix {
    // 矩阵乘法
    public static double[][] multiply(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int rowsB = b.length;
        int colsB = b[0].length;
        if (colsA != rowsB) {
            throw new IllegalArgumentException("矩阵维度不匹配");
        }
        double[][] result = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    // 矩阵转置
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // 将矩阵重塑为 4D 张量
    public static double[][][][] reshapeTo4D(double[][] mat, int batch, int channel, int height, int width) {
        double[][][][] tensor = new double[batch][channel][height][width];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int index = b * channel * height * width + c * height * width + h * width + w;
                        tensor[b][c][h][w] = mat[index / mat[0].length][index % mat[0].length];
                    }
                }
            }
        }
        return tensor;
    }
}    