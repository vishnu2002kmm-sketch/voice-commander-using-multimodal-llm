# ============================================================
#   ARJU COMMANDER — modules/system_commands.py
#
#   FIX — Chrome / Firefox / apps not launching
#   ────────────────────────────────────────────
#   Root cause: subprocess.Popen("chrome.exe") fails because
#   browsers are NOT in the Windows PATH — they install to
#   AppData/Local/Programs.
#   Fix: Use os.system("start <name>") which delegates to
#   Windows' own file association / PATH resolver.
#   For known apps we also probe common install locations.
# ============================================================

import datetime
import glob
import os
import subprocess
import webbrowser
from urllib.parse import quote_plus, urlparse

import config

# ── Optional imports ─────────────────────────────────────────
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    import ctypes
    _AUDIO = True
except ImportError:
    _AUDIO = False

try:
    import screen_brightness_control as sbc
    _BRIGHT = True
except ImportError:
    _BRIGHT = False

try:
    import pyautogui
    _GUI = True
except ImportError:
    _GUI = False

try:
    import pywhatkit as kit
    _KIT = True
except Exception as e:
    print(f"[System] pywhatkit unavailable: {e}")
    _KIT = False


# ── Known Windows app launch commands ─────────────────────────
# Using "start" lets Windows resolve the app without needing
# the exe to be in PATH.
_APPS: dict[str, str] = {
    "notepad":              "notepad",
    "calculator":           "calc",
    "paint":                "mspaint",
    "wordpad":              "wordpad",
    "cmd":                  "cmd",
    "file explorer":        "explorer",
    "explorer":             "explorer",
    "task manager":         "taskmgr",
    "snipping tool":        "snippingtool",
    "settings":             "ms-settings:",
    "word":                 "winword",
    "excel":                "excel",
    "powerpoint":           "powerpnt",
    "vs code":              "code",
    "vscode":               "code",
    "visual studio code":   "code",
    # Browsers — resolved via start command (uses file association)
    "chrome":               "chrome",
    "google chrome":        "chrome",
    "firefox":              "firefox",
    "edge":                 "msedge",
    "microsoft edge":       "msedge",
    # Common apps
    "vlc":                  "vlc",
    "spotify":              "spotify",
    "discord":              "discord",
    "whatsapp":             "whatsapp",
    "telegram":             "telegram",
    "zoom":                 "zoom",
    "teams":                "teams",
}

# Absolute paths to try for browsers (common install locations)
_BROWSER_PATHS = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ],
    "firefox": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    ],
    "msedge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
}


def _find_exe(name: str) -> str | None:
    """Search known paths for an executable."""
    paths = _BROWSER_PATHS.get(name, [])
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def open_app(name: str) -> str:
    """
    Open an application by name.
    Strategy:
    1. If known browser → try absolute path first
    2. Fall back to os.startfile for ms-settings:// URIs
    3. Use subprocess with shell=True + start command
    4. Webbrowser fallback for browsers
    """
    n   = name.lower().strip()
    cmd = _APPS.get(n, n)   # get mapped name or use as-is

    # Settings / URI
    if cmd.startswith("ms-"):
        try:
            os.startfile(cmd)
            return f"Opening {name}."
        except Exception as e:
            return f"Could not open settings: {e}"

    # Try absolute path for browsers
    exe_path = _find_exe(cmd)
    if exe_path:
        try:
            subprocess.Popen([exe_path], shell=False)
            return f"Opening {name}."
        except Exception as e:
            print(f"[System] Direct path failed: {e}")

    # Use Windows 'start' command — resolves PATH + file associations
    try:
        subprocess.Popen(f'start {cmd}', shell=True)
        return f"Opening {name}."
    except Exception:
        pass

    # Browser fallback — open via webbrowser module
    if n in ("chrome", "google chrome", "firefox", "edge", "microsoft edge"):
        webbrowser.open("https://www.google.com")
        return f"Opening {name} via browser fallback."

    return f"Could not open {name}. Make sure it is installed."


def close_app(name: str) -> str:
    n   = name.lower().strip()
    cmd = _APPS.get(n, n)
    exe = cmd if cmd.endswith(".exe") else cmd + ".exe"
    try:
        subprocess.run(["taskkill", "/F", "/IM", exe],
                       capture_output=True, shell=False)
        return f"Closed {name}."
    except Exception as e:
        return f"Could not close {name}: {e}"


# ── Time / Date ───────────────────────────────────────────────

def get_time() -> str:
    return f"The time is {datetime.datetime.now().strftime('%I:%M %p')}."

def get_date() -> str:
    return f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."


# ── Web ───────────────────────────────────────────────────────

def search_google(q: str) -> str:
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(q)}")
    return f"Searching Google for: {q}."

def open_youtube(q: str = None) -> str:
    if q:
        webbrowser.open(
            f"https://www.youtube.com/results?search_query={quote_plus(q)}"
        )
        return f"Searching YouTube for: {q}."
    webbrowser.open("https://www.youtube.com")
    return "Opening YouTube."

def play_youtube(song: str) -> str:
    if _KIT:
        try:
            kit.playonyt(song)
            return f"Playing {song} on YouTube."
        except Exception:
            pass
    return open_youtube(song)

def open_gmail() -> str:
    webbrowser.open("https://mail.google.com")
    return "Opening Gmail."

def open_whatsapp_web() -> str:
    webbrowser.open("https://web.whatsapp.com")
    return "Opening WhatsApp Web."

def open_url(url: str) -> str:
    target = (url or "").strip()
    if not target:
        return "No website was provided."
    if " " in target and "." not in target:
        return search_google(target)
    parsed = urlparse(target if "://" in target else f"https://{target}")
    if not parsed.netloc:
        return search_google(target)
    final_url = parsed.geturl()
    webbrowser.open(final_url)
    return f"Opening {final_url}."

def play_on_site(query: str, site: str = "youtube") -> str:
    q = (query or "").strip()
    s = (site or "youtube").lower().strip()
    if not q:
        return "What should I play?"

    if "youtube" in s:
        return play_youtube(q)
    if "spotify" in s:
        webbrowser.open(f"https://open.spotify.com/search/{quote_plus(q)}")
        return f"Searching Spotify for: {q}."
    if "soundcloud" in s:
        webbrowser.open(f"https://soundcloud.com/search?q={quote_plus(q)}")
        return f"Searching SoundCloud for: {q}."
    if "google" in s:
        return search_google(q)

    webbrowser.open(f"https://www.google.com/search?q={quote_plus(q + ' site:' + s)}")
    return f"Searching {site} for: {q}."


# ── Volume ────────────────────────────────────────────────────

def _vol():
    if not _AUDIO:
        return None
    try:
        iface = AudioUtilities.GetSpeakers().Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        return ctypes.cast(iface, ctypes.POINTER(IAudioEndpointVolume))
    except Exception:
        return None

def increase_volume(step: int = 10) -> str:
    v = _vol()
    if v:
        c  = v.GetMasterVolumeLevelScalar() * 100
        nv = min(100, c + step)
        v.SetMasterVolumeLevelScalar(nv / 100, None)
        return f"Volume increased to {int(nv)} percent."
    # nircmd fallback
    os.system("nircmd.exe changesysvolume 6553")
    return "Volume increased."

def decrease_volume(step: int = 10) -> str:
    v = _vol()
    if v:
        c  = v.GetMasterVolumeLevelScalar() * 100
        nv = max(0, c - step)
        v.SetMasterVolumeLevelScalar(nv / 100, None)
        return f"Volume decreased to {int(nv)} percent."
    os.system("nircmd.exe changesysvolume -6553")
    return "Volume decreased."

def mute_volume() -> str:
    v = _vol()
    if v:
        v.SetMute(1, None)
    else:
        os.system("nircmd.exe mutesysvolume 1")
    return "Muted."

def unmute_volume() -> str:
    v = _vol()
    if v:
        v.SetMute(0, None)
    else:
        os.system("nircmd.exe mutesysvolume 0")
    return "Unmuted."

def set_volume(level: int) -> str:
    level = max(0, min(100, level))
    v = _vol()
    if v:
        v.SetMasterVolumeLevelScalar(level / 100, None)
    return f"Volume set to {level} percent."


# ── Brightness ────────────────────────────────────────────────

def increase_brightness(step: int = 10) -> str:
    if _BRIGHT:
        try:
            c = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(min(100, c + step), display=0)
            return f"Brightness increased to {min(100, c+step)} percent."
        except Exception as e:
            return f"Brightness error: {e}"
    return "Install screen-brightness-control for brightness control."

def decrease_brightness(step: int = 10) -> str:
    if _BRIGHT:
        try:
            c = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(max(0, c - step), display=0)
            return f"Brightness decreased to {max(0, c-step)} percent."
        except Exception as e:
            return f"Brightness error: {e}"
    return "Install screen-brightness-control for brightness control."


# ── Screenshot ────────────────────────────────────────────────

def screenshot() -> str:
    if not _GUI:
        return "Install pyautogui for screenshots."
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    fn   = f"shot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(config.SCREENSHOT_DIR, fn)
    try:
        pyautogui.screenshot().save(path)
        return f"Screenshot saved as {fn}."
    except Exception as e:
        return f"Screenshot failed: {e}"


# ── WhatsApp ─────────────────────────────────────────────────

def send_whatsapp(contact: str, message: str) -> str:
    if not _KIT:
        return "Install pywhatkit for WhatsApp messaging."
    try:
        now = datetime.datetime.now()
        kit.sendwhatmsg(
            contact, message,
            now.hour, (now.minute + 2) % 60,
            wait_time=20
        )
        return f"WhatsApp message scheduled to {contact}."
    except Exception as e:
        return f"WhatsApp error: {e}"


# ── Power ─────────────────────────────────────────────────────

def shutdown() -> str:
    subprocess.run("shutdown /s /t 30", shell=True)
    return "Shutting down in 30 seconds."

def restart() -> str:
    subprocess.run("shutdown /r /t 30", shell=True)
    return "Restarting in 30 seconds."

def lock() -> str:
    subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
    return "Screen locked."
