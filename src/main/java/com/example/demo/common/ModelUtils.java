package com.example.demo.common;

import com.example.demo.face.Layer;
import com.example.demo.service.CNNModel;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

// 模型工具类，包含保存和加载模型的方法
public class ModelUtils {
    // 保存模型
    public static void saveModel(CNNModel model, String path) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
        // 使用公共方法获取 layers
        oos.writeObject(model.getLayers());
        oos.close();
    }

    // 加载模型
    public static CNNModel loadModel(String path) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
        List<Layer> layers = (List<Layer>) ois.readObject();
        CNNModel model = new CNNModel();
        // 使用公共方法设置 layers
        model.setLayers(layers);
        return model;
    }
}