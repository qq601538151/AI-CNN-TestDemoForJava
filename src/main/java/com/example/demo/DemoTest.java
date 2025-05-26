package com.example.demo;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * MNIST手写数字识别测试类
 */
public class DemoTest {
    static {
        // 全局设置为中文环境
        Locale.setDefault(Locale.CHINA);
    }
    public static void main(String[] args) throws Exception {
        // 配置参数
        int batchSize = 64;      // 批次大小
        int seed = 12345;       // 随机种子
        int numEpochs = 5;      // 训练轮数
        int numInputs = 784;    // 输入维度 (28x28像素)
        int numOutputs = 10;    // 输出类别数 (0-9)
        int numHiddenNodes = 100; // 隐藏层节点数
        double learningRate = 0.01; // 学习率
        // 数据路径 (如果为空则自动下载到~/.deeplearning4j/mnist)
        // 指定本地数据路径 (可选)
        String dataPath = "src/main/resources/MNIST_data";
        System.setProperty("dl4j.mnist.data.dir", "src/main/resources/MNIST_data");

        // 加载训练数据
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, seed);

        // 加载测试数据
        DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, seed);
        // MnistDataSetIterator.setLocalFilesLocation(new File(dataPath));

        // 配置神经网络
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .build();

        // 构建并初始化模型
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // 每10次迭代打印分数

        // 训练模型
        System.out.println("开始训练模型...");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
            System.out.println("完成第 " + (i + 1) + " 轮训练");
            trainIter.reset(); // 重置迭代器
        }

        // 评估模型
        System.out.println("评估模型...");
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());

        System.out.println("MNIST测试完成！");
    }
}