package org.lijian.socket;



import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import jakarta.websocket.OnClose;
import jakarta.websocket.OnError;
import jakarta.websocket.OnMessage;
import jakarta.websocket.OnOpen;
import jakarta.websocket.Session;
import jakarta.websocket.server.PathParam;
import jakarta.websocket.server.ServerEndpoint;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;

/**
 * @Author:Long
 **/
@Component
@Slf4j
@ServerEndpoint(value = "/audio/{roomId}/{userId}")
public class WebSocketAudioServer {

    private static ConcurrentHashMap<String, Session> sessionPool = new ConcurrentHashMap<String, Session>();
    private static CopyOnWriteArraySet<WebSocketAudioServer> webSocketSet = new CopyOnWriteArraySet<>();
    private Session webSocketsession;
    private String roomId;
    private String userId;

    @OnOpen
    public void onOpen(@PathParam(value = "roomId") String roomId, @PathParam(value = "userId") String userId,
                       Session webSocketsession) {
        // 接收到发送消息的人员编号
        this.roomId = roomId;
        this.userId = userId;
        // 加入map中，绑定当前用户和socket
        sessionPool.put(userId, webSocketsession);
        webSocketSet.add(this);
        this.webSocketsession = webSocketsession;
        // 在线数加1
//        addOnlineCount();
        System.out.println("user编号:" + userId + "：加入Room:" + roomId + "语音聊天  " + "总数为:" + webSocketSet.size());
    }

    @OnClose
    public void onClose() {
        try {
            sessionPool.remove(this.userId);
        } catch (Exception e) {
        }
    }


    @OnMessage(maxMessageSize = 5242880)
    public void onMessage(@PathParam(value = "roomId") String roomId, @PathParam(value = "userId") String userId,
                          String inputStream) {
        try {

            for (WebSocketAudioServer webSocket : webSocketSet) {
                try {
                    if (webSocket.webSocketsession.isOpen() && webSocket.roomId.equals(roomId)
                            && !webSocket.userId.equals(userId)) {
                        webSocket.webSocketsession.getBasicRemote().sendText(inputStream);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    @OnError
    public void onError(Session session, Throwable error) {
        error.printStackTrace();
    }

    /**
     * 为指定用户发送消息
     *
     */
    public void sendMessage(String message) throws IOException {
        // 加同步锁，解决多线程下发送消息异常关闭
        synchronized (this.webSocketsession) {
            this.webSocketsession.getBasicRemote().sendText(message);
        }
    }

    public List<String> getOnlineUser(String roomId) {
        List<String> userList = new ArrayList<String>();
        for (WebSocketAudioServer webSocketAudioServer : webSocketSet) {
            try {
                if (webSocketAudioServer.webSocketsession.isOpen() && webSocketAudioServer.roomId.equals(roomId)) {
                    if (!userList.contains(webSocketAudioServer.userId)) {
                        userList.add(webSocketAudioServer.userId);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return userList;
    }
}

