package org.lijian.service;

import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.io.File;

@Service
public class SpeechRecognitionService {

    private OnlineRecognizer recognizer;
    private OnlineStream stream;

    @PostConstruct
    public void init() {
        // 配置动态库路径
        try {
            System.load(new File("libs/libsherpa-onnx-jni.so").getAbsolutePath());
            // Windows使用下面这行
            // System.load(new File("libs/sherpa-onnx-jni.dll").getAbsolutePath());

            // 配置模型
            String encoder = "models/encoder-epoch-99-avg-1.int8.onnx";
            String decoder = "models/decoder-epoch-99-avg-1.onnx";
            String joiner = "models/joiner-epoch-99-avg-1.onnx";
            String tokens = "models/tokens.txt";

            OnlineTransducerModelConfig transducer = OnlineTransducerModelConfig.builder()
                    .setEncoder(encoder)
                    .setDecoder(decoder)
                    .setJoiner(joiner)
                    .build();

            OnlineModelConfig modelConfig = OnlineModelConfig.builder()
                    .setTransducer(transducer)
                    .setTokens(tokens)
                    .setNumThreads(1)
                    .setDebug(true)
                    .build();

            OnlineRecognizerConfig config = OnlineRecognizerConfig.builder()
                    .setOnlineModelConfig(modelConfig)
                    .setDecodingMethod("greedy_search")
                    .build();

            recognizer = new OnlineRecognizer(config);
            stream = recognizer.createStream();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 处理音频数据并返回识别结果
     */
    public String processAudio(byte[] audioData) {
        try {
            // 将字节数组转换为float数组
            float[] samples = convertBytesToFloats(audioData);

            // 将样本传递给识别器
            stream.acceptWaveform(samples, 16000);

            // 检查是否准备好解码
            if (recognizer.isReady(stream)) {
                recognizer.decode(stream);
            }

            // 获取识别结果
            String text = recognizer.getResult(stream).getText();

            // 检查是否到达端点，若是则重置流
            if (recognizer.isEndpoint(stream)) {
                recognizer.reset(stream);
            }

            return text;
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    /**
     * 将字节数组转换为浮点数组（16位PCM音频格式）
     */
    private float[] convertBytesToFloats(byte[] audioData) {
        float[] samples = new float[audioData.length / 2];
        for (int i = 0; i < samples.length; i++) {
            short s = (short) ((audioData[i * 2 + 1] & 0xff) << 8 | (audioData[i * 2] & 0xff));
            samples[i] = s / 32768.0f;
        }
        return samples;
    }

    @PreDestroy
    public void cleanup() {
        if (stream != null) {
            stream.release();
        }
        if (recognizer != null) {
            recognizer.release();
        }
    }
}