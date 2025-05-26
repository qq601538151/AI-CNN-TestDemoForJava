# AI-CNN-TestDemoForJava
AI-CNN-TestDemoForJava
整体大纲如下，分为两个测试类，一个目的跑通代码，另外一个应用公开数据集修改测试
![image](https://github.com/user-attachments/assets/70739d02-e513-4191-a46e-02619c5f5d1b)
## 初始测试结果



Epoch 1, Loss: 0.111944
Epoch 2, Loss: 0.111944
Epoch 3, Loss: 0.111944
Epoch 4, Loss: 0.111944
Epoch 5, Loss: 0.111944

## MNIST数据集测试结果

使用jar包 DeepLearning4j ---  1.0.0-beta7版本  jdk17 框架spring-boot 3.1.5

jar包pom依赖

```xml
     <properties>
        <java.version>17</java.version>
        <deeplearning4j.version>1.0.0-beta7</deeplearning4j.version>
    </properties>
    
    
 <!-- Deeplearning4j 依赖 -->
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native-platform</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-api</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-local</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-data-image</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-datasets</artifactId>
            <version>${deeplearning4j.version}</version>
        </dependency>
```

### 结果

```
开始训练模型...
09:53:36.210 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 0 is 2.3097667694091797
09:53:36.380 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 10 is 2.2330074310302734
09:53:36.523 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 20 is 2.148519992828369
09:53:36.598 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 30 is 1.9896578788757324
09:53:36.682 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 40 is 1.9534538984298706
09:53:36.770 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 50 is 1.8600245714187622
09:53:36.834 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 60 is 1.9147177934646606
09:53:36.898 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 70 is 1.835957407951355
09:53:36.945 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 80 is 1.6288288831710815
09:53:37.000 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 90 is 1.621422529220581
---------
----------
-----------
09:53:51.409 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4630 is 0.37666547298431396
09:53:51.430 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4640 is 0.3868129551410675
09:53:51.455 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4650 is 0.2551277279853821
09:53:51.483 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4660 is 0.1291111707687378
09:53:51.510 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4670 is 0.22649961709976196
09:53:51.540 [main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener -- Score at iteration 4680 is 0.2791060507297516
完成第 5 轮训练

```

### F1评分

```
========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0.9216
 Precision:       0.9212
 Recall:          0.9204
 F1 Score:        0.9204
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)
```

### 混淆矩阵

```
=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9
---------------------------------------------------
  965    0    0    2    0    3    7    1    2    0 | 0 = 0
    0 1110    2    3    1    3    4    1   11    0 | 1 = 1
   11    5  927   17   15    0   10   13   30    4 | 2 = 2
    3    2   18  937    0   16    2   14   12    6 | 3 = 3
    1    1    3    1  924    0   11    2    5   34 | 4 = 4
   11    3    4   47    8  768   18    5   21    7 | 5 = 5
   16    3    5    0   17   10  903    1    3    0 | 6 = 6
    3   11   25    8    9    0    0  944    2   26 | 7 = 7
    9    9    8   32   11   21   15   15  844   10 | 8 = 8
   14    7    3   11   43    5    1   27    4  894 | 9 = 9

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

