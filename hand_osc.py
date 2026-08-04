import cv2
import mediapipe as mp
import threading
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pythonosc import udp_client

# --- 設定 ---
LERP_ALPHA = 0.2        # 座標補間係数
RENDER_TARGET_FPS = 60  # 描画目標FPS

# --- OSC設定 ---
OSC_IP = "127.0.0.1"    # 送信先IP
OSC_PORT = 8002         # 送信先ポート
SEND_INTERVAL = 0.1     # 通信頻度

# --- データ構造 ---
@dataclass
class Point3D:
    x: float
    y: float
    z: float

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame: Optional[cv2.Mat] = None
        self.target_landmarks: Optional[List[Point3D]] = None
        self.inference_fps = 0.0

# --- 魔法陣描画関数 (色指定に対応) ---
def draw_magic_circle(img, x, y, radius, time_sec, color):
    """
    指定された位置・サイズ・色で魔法陣を描画する
    """
    if radius < 1: return

    # 1. 回転する正方形 (指定色)
    angle = (time_sec * 90) % 360
    pts = []
    for i in range(4):
        theta = math.radians(angle + i * 90)
        px = int(x + radius * 1.0 * math.cos(theta))
        py = int(y + radius * 1.0 * math.sin(theta))
        pts.append([px, py])
    cv2.polylines(img, [np.array(pts)], True, color, 2, cv2.LINE_AA)
    
    # 2. 逆回転する四角形 (白でアクセント)
    angle2 = -(time_sec * 180) % 360
    pts2 = []
    for i in range(4):
        theta = math.radians(angle2 + i * 90)
        px = int(x + radius * 1.0 * math.cos(theta))
        py = int(y + radius * 1.0 * math.sin(theta))
        pts2.append([px, py])
    cv2.polylines(img, [np.array(pts2)], True, color, 1, cv2.LINE_AA)
    cv2.polylines(img, [np.array(pts2)], True, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 3. 外周と中心
    cv2.circle(img, (x, y), int(radius * 1.2), color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

# --- ユーティリティ関数 ---
def lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha

def convert_landmarks_to_points(mp_landmarks) -> List[Point3D]:
    return [Point3D(lm.x, lm.y, lm.z) for lm in mp_landmarks.landmark]

def calculate_distance(p1: Point3D, p2: Point3D) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- 推論スレッド ---
def inference_worker(shared_state: SharedState):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    while shared_state.running and cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        with shared_state.lock:
            shared_state.latest_frame = frame
            shared_state.inference_fps = fps
            if results.multi_hand_landmarks:
                shared_state.target_landmarks = convert_landmarks_to_points(results.multi_hand_landmarks[0])
            else:
                shared_state.target_landmarks = None

    cap.release()
    hands.close()

# --- メインループ ---
def main():
    shared_state = SharedState()
    thread = threading.Thread(target=inference_worker, args=(shared_state,), daemon=True)
    thread.start()

    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    last_send_time = 0
    
    print(f"OSC Target: {OSC_IP}:{OSC_PORT}")
    print("Sending distances for Index, Middle, Ring, Little fingers.")

    current_landmarks: Optional[List[Point3D]] = None
    mp_hands = mp.solutions.hands
    hand_connections = mp_hands.HAND_CONNECTIONS
    
    # 各指の設定
    fingers_map = [
        {"name": "/index",  "idx": 8,  "color": (0, 255, 255)}, # 黄
        {"name": "/middle", "idx": 12, "color": (0, 255, 0)},   # 緑
        {"name": "/ring",   "idx": 16, "color": (255, 100, 0)}, # 青
        {"name": "/little", "idx": 20, "color": (255, 0, 255)}, # 紫
    ]

    last_loop_time = time.time()
    print("Starting... Press 'q' to exit.")
    
    try:
        while True:
            loop_start_time = time.time()
            last_loop_time = loop_start_time

            # データ取得
            frame_to_draw = None
            target_landmarks = None
            
            with shared_state.lock:
                if shared_state.latest_frame is not None:
                    frame_to_draw = shared_state.latest_frame.copy()
                target_landmarks = shared_state.target_landmarks

            if frame_to_draw is None:
                time.sleep(0.01)
                continue

            h, w, _ = frame_to_draw.shape

            # 座標補間
            if target_landmarks is not None:
                if current_landmarks is None:
                    current_landmarks = target_landmarks
                else:
                    for i in range(len(current_landmarks)):
                        current_landmarks[i].x = lerp(current_landmarks[i].x, target_landmarks[i].x, LERP_ALPHA)
                        current_landmarks[i].y = lerp(current_landmarks[i].y, target_landmarks[i].y, LERP_ALPHA)
                        current_landmarks[i].z = lerp(current_landmarks[i].z, target_landmarks[i].z, LERP_ALPHA)
            else:
                current_landmarks = None

            # 描画 & 処理
            if current_landmarks is not None:
                wrist = current_landmarks[0]
                middle_mcp = current_landmarks[9]
                thumb_tip = current_landmarks[4]

                # 手の大きさを計算（基準として使用）
                hand_size_px = math.sqrt(((wrist.x - middle_mcp.x) * w)**2 + ((wrist.y - middle_mcp.y) * h)**2)
                # 指先の魔法陣サイズ（手の大きさの10%程度）
                tip_radius = int(hand_size_px * 0.1)

                # 1. 骨格線を描画
                for connection in hand_connections:
                    p1 = current_landmarks[connection[0]]
                    p2 = current_landmarks[connection[1]]
                    cv2.line(frame_to_draw, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 255, 255), 1)

                # 2. 親指の魔法陣 (赤色)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                draw_magic_circle(frame_to_draw, thumb_x, thumb_y, tip_radius, loop_start_time, (0, 0, 255))

                should_send = (loop_start_time - last_send_time > SEND_INTERVAL)

                # 3. 各指の処理 (魔法陣 + 距離線 + OSC)
                for finger in fingers_map:
                    target_tip = current_landmarks[finger["idx"]]
                    
                    # 座標
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h) # 親指
                    fx, fy = int(target_tip.x * w), int(target_tip.y * h) # 各指

                    # 距離計算
                    dist = calculate_distance(thumb_tip, target_tip)
                    
                    # 指先に魔法陣を描画
                    draw_magic_circle(frame_to_draw, fx, fy, tip_radius, loop_start_time, finger["color"])

                    # 視覚化ライン (親指と各指を結ぶ)
                    cv2.line(frame_to_draw, (tx, ty), (fx, fy), finger["color"], 2)
                    
                    # 距離テキスト
                    mid_x, mid_y = (tx + fx) // 2, (ty + fy) // 2
                    cv2.putText(frame_to_draw, f"{dist:.2f}", (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger["color"], 1)

                    # OSC送信
                    if should_send:
                        osc_client.send_message(finger["name"], dist)

                if should_send:
                    last_send_time = loop_start_time

            cv2.imshow('Finger Tip Magic Circles', frame_to_draw)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        shared_state.running = False
        thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()