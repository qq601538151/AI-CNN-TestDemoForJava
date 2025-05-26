package com.example.demo.service;

import com.example.demo.common.Tensor;

import java.util.HashMap;
import java.util.Map;

// Adam 优化器实现
public class AdamOptimizer {
    private double lr;
    private double beta1;
    private double beta2;
    private double eps;
    private int t;
    private Map<String, Tensor> v;
    private Map<String, Tensor> s;

    // 构造函数，初始化优化器参数
    public AdamOptimizer(double lr, double beta1, double beta2, double eps) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.t = 0;
        this.v = new HashMap<>();
        this.s = new HashMap<>();
    }

    // 更新参数
    public void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        t++;
        for (Map.Entry<String, Tensor> entry : params.entrySet()) {
            String key = entry.getKey();
            Tensor param = entry.getValue();
            Tensor g = grads.get(key);

            if (!v.containsKey(key)) {
                v.put(key, new Tensor(param.getBatchSize(), param.getDepth(), param.getHeight(), param.getWidth()));
                s.put(key, new Tensor(param.getBatchSize(), param.getDepth(), param.getHeight(), param.getWidth()));
            }

            v.put(key, v.get(key).multiply(beta1).add(g.multiply(1 - beta1)));
            s.put(key, s.get(key).multiply(beta2).add(g.square().multiply(1 - beta2)));

            Tensor vCorrected = v.get(key).divide(1 - Math.pow(beta1, t));
            Tensor sCorrected = s.get(key).divide(1 - Math.pow(beta2, t));

            param = param.subtract(vCorrected.multiply(lr).divide(sCorrected.sqrt().add(eps)));
            params.put(key, param);
        }
    }
}    