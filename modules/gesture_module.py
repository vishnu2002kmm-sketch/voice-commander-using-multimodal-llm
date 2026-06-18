# ============================================================
#   VOICE COMMANDER — modules/gesture_module.py
# ============================================================

# ============================================================
#   VOICE COMMANDER — modules/gesture_module.py
# ============================================================
import threading, time
import config

try:
    import cv2
    import mediapipe as mp
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as draw
    _MP = True
except Exception as e:
    print(f"\n[GESTURE MODULE ERROR] Failed to load dependencies:")
    print(f"Details: {e}\n")
    _MP = False  # <--- Fixed the 'Falses' typo here!

from modules import system_commands as sc

class GestureController:
    def __init__(self, voice=None):
        self.voice = voice; self._running = False; self._muted = False

    @property
    def available(self): return _MP

    def _classify(self, lm) -> str:
        tips=[8,12,16,20]; pips=[6,10,14,18]
        up = [t for t,p in zip(tips,pips) if lm[t].y < lm[p].y]
        if lm[4].x < lm[3].x and not up: return "THUMB_UP"
        if lm[4].y > lm[3].y and not up: return "THUMB_DOWN"
        if len(up) >= 4:                  return "OPEN_PALM"
        if not up and not (lm[4].x < lm[3].x): return "FIST"
        return "NONE"

    def _act(self, g):
        msg = ""
        if   g=="THUMB_UP":   msg=sc.increase_volume(config.GESTURE_VOL_STEP)
        elif g=="THUMB_DOWN": msg=sc.decrease_volume(config.GESTURE_VOL_STEP)
        elif g=="OPEN_PALM":
            msg=sc.unmute_volume() if self._muted else sc.mute_volume()
            self._muted=not self._muted
        elif g=="FIST": self.stop(); msg="Gesture control stopped."
        if msg:
            print(f"[Gesture] {g} → {msg}")
            if self.voice: self.voice.speak(msg)

    def _loop(self):
        cap   = cv2.VideoCapture(config.CAM_INDEX)
        
        hands_model = mp_hands.Hands(
            model_complexity=0, min_detection_confidence=0.7,
            min_tracking_confidence=0.5, max_num_hands=1)
            
        last_g, last_t, cd = None, 0, 1.5
        with hands_model as hands:
            while self._running:
                ok, frame = cap.read()
                if not ok: break
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                g   = "NONE"
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        g = self._classify(hl.landmark)
                now = time.time()
                if g!="NONE" and g!=last_g and now-last_t>cd:
                    self._act(g); last_g=g; last_t=now
                cv2.putText(frame, f"Gesture: {g}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, "Q=stop", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                cv2.imshow("Gesture Control", frame)
                if cv2.waitKey(1)&0xFF==ord('q'): self._running=False
        cap.release(); cv2.destroyAllWindows()

    def start(self) -> str:
        if not _MP:       return "Install mediapipe and opencv-python."
        if self._running: return "Gesture control already running."
        self._running=True
        threading.Thread(target=self._loop, daemon=True).start()
        return "Gesture control started. Show your hand. Q or fist to stop."

    def stop(self) -> str:
        self._running=False; return "Gesture control stopped."