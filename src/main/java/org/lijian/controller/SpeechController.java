package org.lijian.controller;

import org.lijian.service.SpeechRecognitionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
public class SpeechController {

    @Autowired
    private SpeechRecognitionService speechService;

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    /**
     * 处理从客户端接收的音频数据
     * 客户端通过/app/speech发送消息
     */
    @MessageMapping("/speech")
    public void processAudio(byte[] audioData) {
        // 将音频数据传递给语音识别服务
        String recognizedText = speechService.processAudio(audioData);

        // 如果识别出文本，发送给订阅了/topic/transcription的客户端
        if (recognizedText != null && !recognizedText.isEmpty()) {
            messagingTemplate.convertAndSend("/topic/transcription", recognizedText);
        }
    }
}