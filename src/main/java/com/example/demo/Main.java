package com.example.demo;

import com.example.demo.common.Tensor;
import com.example.demo.face.Layer;
import com.example.demo.service.AdamOptimizer;
import com.example.demo.service.CNNModel;
import com.example.demo.service.CrossEntropyLoss;
import com.example.demo.service.SimpleLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
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

        // 加载MNIST数据集
        DataSetIterator mnistTrain = loadMNISTData("train", batchSize, inputChannels, inputHeight, inputWidth, numClasses);
        DataSetIterator mnistTest = loadMNISTData("test", batchSize, inputChannels, inputHeight, inputWidth, numClasses);

        // 训练模型
        for (int epoch = 0; epoch < epochs; epoch++) {
            while (mnistTrain.hasNext()) {
                DataSet ds = mnistTrain.next();
                Tensor input = convertToTensor(ds.getFeatures());
                Tensor target = convertToTensor(ds.getLabels());
                model.trainStep(input, target);
            }
            mnistTrain.reset();

            // 计算损失
            double totalLoss = 0;
            int numBatches = 0;
            while (mnistTest.hasNext()) {
                DataSet ds = mnistTest.next();
                Tensor input = convertToTensor(ds.getFeatures());
                Tensor target = convertToTensor(ds.getLabels());
                double loss = lossFunction.compute(model.forward(input, false), target);
                totalLoss += loss;
                numBatches++;
            }
            mnistTest.reset();
            double averageLoss = totalLoss / numBatches;
            System.out.printf("Epoch %d, Loss: %.6f%n", epoch + 1, averageLoss);
        }
    }

    private static DataSetIterator loadMNISTData(String type, int batchSize, int inputChannels, int inputHeight, int inputWidth, int numClasses) throws IOException {
        String basePath = System.getProperty("user.dir") + "/src/main/resources/MNIST_data/";
        String imagesFile;
        String labelsFile;
        if ("train".equals(type)) {
            imagesFile = basePath + "train-images-idx3-ubyte.gz";
            labelsFile = basePath + "train-labels-idx1-ubyte.gz";
        } else {
            imagesFile = basePath + "t10k-images-idx3-ubyte.gz";
            labelsFile = basePath + "t10k-labels-idx1-ubyte.gz";
        }

        org.nd4j.linalg.dataset.api.preprocessor.DataNormalization scaler = new org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler(0, 1);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator = new org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator(batchSize, true, 12345);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);
        return iterator;
    }

    private static Tensor convertToTensor(INDArray ndArray) {
        int batchSize = (int)ndArray.size(0);
        int channels = 1;
        int height = 28;
        int width = 28;
        Tensor tensor = new Tensor(batchSize, channels, height, width);
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double value = ndArray.getDouble(b, h * width + w);
                    tensor.setElement(b, 0, h, w, value);
                }
            }
        }
        return tensor;
    }
}