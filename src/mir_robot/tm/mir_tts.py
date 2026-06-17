import sys
import time
import requests
import base64
import os
import json

try:
    from gtts import gTTS
except ImportError:
    print("Vui lòng chạy: pip install gTTS")
    sys.exit(1)

# Có thể tận dụng file navigationcacdiem.py có sẵn
import navigationcacdiem as nav


def generate_tts_wav(text: str, output_wav_path: str):
    """
    Sử dụng gTTS (Google Text-to-Speech)
    """
    # 1. Xử lý các từ tiếng Anh để Google dễ đọc
    text_fixed = text.replace("order", "o đờ").replace("coca", "cô ca").replace("lavie", "la vi").replace("Menu", "me nu").replace("timeout", "thai mao").replace("reset", "ri sét")
    print(f"[TTS] Đang tổng hợp giọng nói Google cho: '{text_fixed}'...")
    
    # 2. Sinh giọng nói Google
    tts = gTTS(text=text_fixed, lang='vi', slow=False)
    
    # Ghi ra file (gTTS mặc định là mp3, nhưng MiR có thể đọc được base64 của file này)
    tts.save(output_wav_path)
    print(f"[TTS] Đã lưu file âm thanh (Google Voice) tại: {output_wav_path}")


def upload_sound_to_mir(headers: dict, wav_path: str, sound_name: str) -> str:
    """
    Upload file WAV lên MiR (base64 encoded) và trả về sound_guid
    """
    print(f"[MiR] Đang tải {wav_path} lên MiR ...")
    
    with open(wav_path, "rb") as audio_file:
         encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
         
    payload = {
        "name": sound_name,
        "sound": encoded_string, 
        "volume": 80
    }
    
    # 1. Kiểm tra xem sound name này đã tồn tại chưa để xoá đi (tránh trùng lặp đầy thẻ nhớ)
    try:
        r = requests.get(f"{nav.API_URL}/sounds", headers=headers, timeout=5)
        for s in r.json():
            if s.get("name") == sound_name:
                requests.delete(f"{nav.API_URL}/sounds/{s['guid']}", headers=headers, timeout=3)
                print(f"[MiR] Đã xoá sound cũ: {sound_name}")
    except Exception:
        pass

    # 2. Upload mới
    resp = requests.post(f"{nav.API_URL}/sounds", json=payload, headers=headers, timeout=10)
    if resp.status_code not in (200, 201):
        raise Exception(f"Upload sound failed: {resp.status_code} - {resp.text}")
    
    sound_guid = resp.json()["guid"]
    print(f"[MiR] Tải lên thành công! Sound GUID: {sound_guid}")
    return sound_guid


def play_sound_on_mir(headers: dict, sound_guid: str, volume: int = 100):
    """
    Kích hoạt phát âm thanh sử dụng hệ thống Mission Queue
    """
    print("[MiR] Khởi tạo lệnh phát âm thanh (Play Sound)...")
    
    # 1. Tạo một Mission ảo (chỉ tồn tại để chứa action Play Sound)
    mission_payload = {
        "name": "TTS_Play_Sound_Temp",
        "group_id": "mirconst-guid-0000-0001-missiongroup" 
    }
    resp = requests.post(f"{nav.API_URL}/missions", json=mission_payload, headers=headers, timeout=5)
    if resp.status_code not in (200, 201):
         raise Exception(f"Create mission failed: {resp.text}")
    mission_guid = resp.json()["guid"]
    
    # 2. Thêm hành động "sound" vào Mission
    action_payload = {
        "action_type": "sound",
        "parameters": [
            {"id": "sound", "value": sound_guid},
            {"id": "volume", "value": float(volume)},
            {"id": "mode", "value": "full"},
            {"id": "duration", "value": "00:00:00"}
        ],
        "priority": 1
    }
    resp = requests.post(f"{nav.API_URL}/missions/{mission_guid}/actions", json=action_payload, headers=headers, timeout=5)
    if resp.status_code not in (200, 201):
        print(f"[Lỗi] Create action failed: {resp.text}")
        return
        
    # 3. Đẩy Mission này lên cùng lên Mission Queue của MiR
    print("[MiR] Thêm lệnh nói vào Queue ...")
    queue_payload = {"mission_id": mission_guid}
    resp = requests.post(f"{nav.API_URL}/mission_queue", json=queue_payload, headers=headers, timeout=5)
    if resp.status_code not in (200, 201):
        print(f"[Lỗi] Add to queue failed: {resp.text}")
        return
        
    
    # Lưu ý: Khi không cần mission_guid này nữa, ta có thể cleanup (Tuỳ chọn)
    # Tuy nhiên robot không cho xoá mission nếu nó đang chạy trong queue

def speak_on_mir(text: str):
    """Hàm tổng hợp từ A-Z để gọi"""
    headers = nav.api_login()
    if not headers:
        print("[Lỗi] Không thể đăng nhập vào MiR REST API!")
        return
        
    wav_filename = "/tmp/temp_tts.wav"
    
    # B1: Chuyển văn bản thành giọng nói (WAV)
    generate_tts_wav(text, wav_filename)
    
    # B2: Tải lên MiR
    sound_guid = upload_sound_to_mir(headers, wav_filename, "tts_temp_voice")
    
    # B3: Phát qua loa MiR
    play_sound_on_mir(headers, sound_guid)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        noi_dung = " ".join(sys.argv[1:])
    else:
        noi_dung = "Xin chào tôi là rô bốt phục vụ thông minh."
        
    speak_on_mir(noi_dung)
