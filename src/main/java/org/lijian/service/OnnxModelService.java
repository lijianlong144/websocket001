package org.lijian.service;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Service
public class OnnxModelService {
    private OrtEnvironment env;
    private OrtSession session;
    private double accuracy;
    private Map<Integer, String> classMapping = new HashMap<>();

    @Autowired
    private ResourceLoader resourceLoader;

    public OnnxModelService() throws OrtException, IOException {
        // 初始化ONNX Runtime环境
        env = OrtEnvironment.getEnvironment();
        // 从类路径加载模型
        Resource resource = resourceLoader.getResource("classpath:iris_classifier.onnx");
        Path modelPath = resource.getFile().toPath();
        session = env.createSession(modelPath.toString());

        // 从JSON文件中加载类别映射
        Resource mappingResource = resourceLoader.getResource("classpath:class_mapping.json");
        ObjectMapper objectMapper = new ObjectMapper();
        classMapping = objectMapper.readValue(mappingResource.getFile(), HashMap.class);

        // 为了演示，这里硬编码了准确率
        this.accuracy = 1.0; // 如果需要，可以用实际计算的准确率替换

    }

    public float[] predict(float[][] inputData) throws OrtException {
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", inputTensor);

        // 模型的主要处理部分
        OrtSession.Result outputs = session.run(inputs);

        Optional<OnnxValue> optionalValue = outputs.get("output");
        OnnxTensor predictionsTensor = (OnnxTensor) optionalValue.get();
        float[] predictions = (float[]) predictionsTensor.getValue();

        // 清理资源
        inputTensor.close();
        outputs.close();
        return predictions;
    }
}

