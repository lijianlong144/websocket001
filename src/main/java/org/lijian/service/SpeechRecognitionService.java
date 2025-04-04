package org.lijian.service;

import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

@Service
public class SpeechRecognitionService {

    private OnlineRecognizer recognizer;
    private OnlineStream stream;

    @PostConstruct
    public void init() {
        try {
            // 加载动态库
            loadNativeLibrary();

            // 获取模型文件路径
            String rootDir = new File(".").getAbsolutePath();
            Path modelPath = Paths.get(rootDir, "src", "main", "resources", "static", "models",
                    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20");

            // 确保模型目录存在
            if (!Files.exists(modelPath)) {
                throw new RuntimeException("模型目录不存在: " + modelPath);
            }

            String encoder = modelPath.resolve("encoder-epoch-99-avg-1.int8.onnx").toString();
            String decoder = modelPath.resolve("decoder-epoch-99-avg-1.onnx").toString();
            String joiner = modelPath.resolve("joiner-epoch-99-avg-1.onnx").toString();
            String tokens = modelPath.resolve("tokens.txt").toString();

            // 验证模型文件是否存在
            validateModelFiles(encoder, decoder, joiner, tokens);

            // 配置模型
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

            // 创建识别器和流
            recognizer = new OnlineRecognizer(config);
            stream = recognizer.createStream();

            System.out.println("语音识别服务初始化成功!");
        } catch (Exception e) {
            System.err.println("语音识别服务初始化失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 加载本地动态库
     */
    private void loadNativeLibrary() throws IOException {
        // 确定操作系统类型
        String osName = System.getProperty("os.name").toLowerCase();
        String libName;

        if (osName.contains("win")) {
            libName = "sherpa-onnx-jni.dll";
        } else if (osName.contains("mac")) {
            libName = "libsherpa-onnx-jni.dylib";
        } else {
            libName = "libsherpa-onnx-jni.so";
        }

        // 先尝试从资源目录加载
        String rootDir = new File(".").getAbsolutePath();
        Path libPath = Paths.get(rootDir, "src", "main", "resources", "static", "lib", libName);

        if (Files.exists(libPath)) {
            System.load(libPath.toString());
            System.out.println("从资源目录加载动态库成功: " + libPath);
            return;
        }

        // 如果资源目录中没有，尝试从类路径加载
        try (InputStream in = getClass().getClassLoader().getResourceAsStream("lib/" + libName)) {
            if (in != null) {
                File tempLib = File.createTempFile("sherpa-onnx-jni", libName.contains(".") ? libName.substring(libName.lastIndexOf('.')) : "");
                tempLib.deleteOnExit();
                Files.copy(in, tempLib.toPath(), StandardCopyOption.REPLACE_EXISTING);
                System.load(tempLib.getAbsolutePath());
                System.out.println("从类路径加载动态库成功: " + tempLib.getAbsolutePath());
                return;
            }
        }

        // 如果都找不到，尝试从当前目录加载
        Path localLibPath = Paths.get("lib", libName);
        if (Files.exists(localLibPath)) {
            System.load(localLibPath.toString());
            System.out.println("从当前目录加载动态库成功: " + localLibPath);
            return;
        }

        throw new RuntimeException("无法找到动态库: " + libName);
    }

    /**
     * 验证模型文件是否存在
     */
    private void validateModelFiles(String encoder, String decoder, String joiner, String tokens) {
        if (!new File(encoder).exists()) {
            throw new RuntimeException("编码器模型文件不存在: " + encoder);
        }
        if (!new File(decoder).exists()) {
            throw new RuntimeException("解码器模型文件不存在: " + decoder);
        }
        if (!new File(joiner).exists()) {
            throw new RuntimeException("连接器模型文件不存在: " + joiner);
        }
        if (!new File(tokens).exists()) {
            throw new RuntimeException("词表文件不存在: " + tokens);
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