"""
launcher.pyw — [資服版] 視覺助理啟動器
=====================================
取代黑色終端機的小視窗：按「啟動」→ 顯示載入進度 → 準備好後秀出 QR Code，手機一掃就能用。
關掉視窗時伺服器也會一起關。詳細紀錄收在「顯示紀錄」裡。

雙擊執行（.pyw 不會跳出終端機），或用桌面上的「視覺助理」捷徑。
原本的 啟動.bat 保留，啟動器有問題時可以改用它。
"""

import os
import re
import sys
import queue
import socket
import threading
import subprocess
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
ICON = os.path.join(ROOT, "app", "static", "icon-512.png")
PORT = 8443            # 預設連接埠；不能用時伺服器會自動改用下一個
COLW = 400            # 內容欄寬，固定住才不會在秀出 QR Code 時跳版

# 配色和手機頁面一致
BG = "#0a0c12"
CARD = "#16171c"
LINE = "#2a2c33"
FG = "#ffffff"
MUTED = "#9a9ca3"
ACCENT = "#5ac8fa"
OK = "#32d74b"
DANGER = "#ff453a"
FONT = "Microsoft JhengHei UI"

# 伺服器輸出 → 要顯示的進度（依出現順序）
STAGES = [
    ("載入模型", "正在載入影像模型…", 0.15),
    ("載入完成，實際使用", "影像模型載入完成", 0.55),
    ("載入語音辨識", "正在載入語音辨識…", 0.65),
    ("語音辨識載入完成", "語音辨識載入完成", 0.9),
]


def rounded(w, h, r, fill, outline=None):
    """畫圓角矩形圖片（tkinter 本身沒有圓角）。先畫 3 倍大再縮小，邊緣才平滑。"""
    s = 3
    im = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * s - 1, h * s - 1], r * s, fill=fill,
                                         outline=outline, width=2 * s if outline else 0)
    return ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS))


class PillButton(tk.Canvas):
    """圓角大按鈕。"""

    def __init__(self, master, text, command, w=300, h=64):
        super().__init__(master, width=w, height=h, bg=BG, highlightthickness=0, cursor="hand2")
        self.w, self.h, self.command = w, h, command
        self.enabled = True
        self.bind("<Button-1>", lambda e: self.enabled and self.command())
        self.bind("<Enter>", lambda e: self._draw(hover=True))
        self.bind("<Leave>", lambda e: self._draw())
        self.set(text, "primary")

    def set(self, text, style="primary", enabled=True):
        self.text, self.style, self.enabled = text, style, enabled
        self.config(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _draw(self, hover=False):
        self.delete("all")
        fill, fg, outline = {"primary": ("#ffffff", "#000000", None),
                             "danger": (CARD, DANGER, DANGER),
                             "disabled": ("#3a3b40", "#9a9ca3", None)}[self.style]
        if hover and self.enabled and self.style == "primary":
            fill = "#dfe6ee"
        self._img = rounded(self.w, self.h, self.h // 2, fill, outline)
        self.create_image(0, 0, image=self._img, anchor="nw")
        self.create_text(self.w // 2, self.h // 2, text=self.text, fill=fg, font=(FONT, 16, "bold"))


class Launcher:
    def __init__(self):
        try:   # 讓工作列顯示自己的圖示，不要顯示成 Python
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VisualAssistant.Launcher")
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.root = tk.Tk()
        self.root.title("視覺助理")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        icon = Image.open(ICON)
        self._win_icon = ImageTk.PhotoImage(icon.resize((64, 64), Image.LANCZOS))
        self.root.iconphoto(True, self._win_icon)

        self.proc = None
        self.lines = queue.Queue()
        self.stage = 0.0
        self.ready = False
        self.anim = 0

        wrap = tk.Frame(self.root, bg=BG, padx=36, pady=28)
        wrap.pack()
        tk.Frame(wrap, width=COLW, height=0, bg=BG).pack()

        self._logo = ImageTk.PhotoImage(icon.resize((112, 112), Image.LANCZOS))
        tk.Label(wrap, image=self._logo, bg=BG).pack(pady=(4, 10))
        tk.Label(wrap, text="視覺助理", fg=FG, bg=BG, font=(FONT, 24, "bold")).pack()
        tk.Label(wrap, text="手機當眼睛，筆電來辨識", fg=MUTED, bg=BG, font=(FONT, 11)).pack(pady=(2, 20))

        # 狀態列：燈號 + 文字 + 進度條
        st = tk.Frame(wrap, bg=BG)
        st.pack(fill="x")
        self.dot = tk.Canvas(st, width=14, height=14, bg=BG, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.status = tk.Label(st, text="尚未啟動", fg=MUTED, bg=BG, font=(FONT, 12), anchor="w")
        self.status.pack(side="left", fill="x")
        self.bar = tk.Canvas(wrap, width=COLW, height=6, bg=BG, highlightthickness=0)
        self.bar.pack(pady=(10, 18))
        self._set_dot(MUTED)
        self._draw_bar()

        # QR Code 區（準備好才出現）
        self.qr_box = tk.Frame(wrap, bg=BG)
        self.qr_label = tk.Label(self.qr_box, bg=BG)
        self.qr_label.pack()
        self.url_label = tk.Label(self.qr_box, text="", fg=ACCENT, bg=BG, font=(FONT, 10, "bold"),
                                  cursor="hand2")
        self.url_label.pack(pady=(8, 0))
        self.url_label.bind("<Button-1>", lambda e: self.copy_url())
        self.hint = tk.Label(self.qr_box, text="", fg=MUTED, bg=BG, font=(FONT, 10), justify="center")
        self.hint.pack(pady=(4, 0))

        self.btn = PillButton(wrap, "啟動", self.toggle, w=COLW)
        self.btn.pack(pady=(16, 10))

        self.log_toggle = tk.Label(wrap, text="顯示紀錄 ▾", fg=MUTED, bg=BG, font=(FONT, 10), cursor="hand2")
        self.log_toggle.pack()
        self.log_toggle.bind("<Button-1>", lambda e: self.toggle_log())
        self.log = tk.Text(wrap, width=56, height=12, bg=CARD, fg="#c8cad0", insertbackground=FG,
                           relief="flat", font=("Consolas", 9), wrap="char", highlightthickness=1,
                           highlightbackground=LINE)
        self.log_shown = False

        self.url = ""
        if self.port_in_use():
            self.set_status("已經有一個伺服器在執行（可能是另一個視窗），請先關掉它", DANGER)
        self.root.after(100, self.poll)

    # ---------------- 狀態顯示 ----------------
    def _set_dot(self, color):
        self.dot.delete("all")
        self.dot.create_oval(2, 2, 12, 12, fill=color, outline="")

    def set_status(self, text, color=MUTED):
        self.status.config(text=text, fg=FG if color in (OK, ACCENT) else color)
        self._set_dot(color)

    def _draw_bar(self):
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, COLW, 6, fill=LINE, outline="")
        if self.proc and not self.ready:
            w = max(12, int(COLW * self.stage))
            self.bar.create_rectangle(0, 0, w, 6, fill=ACCENT, outline="")
            x = (self.anim * 6) % (w + 60) - 60          # 跑動的光點，表示還在處理
            self.bar.create_rectangle(max(0, x), 0, min(w, x + 60), 6, fill="#a6e3ff", outline="")
        elif self.ready:
            self.bar.create_rectangle(0, 0, COLW, 6, fill=OK, outline="")

    # ---------------- 啟動 / 停止 ----------------
    def toggle(self):
        if self.proc:
            self.stop()
        else:
            self.start()

    def start(self):
        if not os.path.exists(PYTHON):
            self.set_status("找不到 .venv 虛擬環境，請先依 README_資服版.md 安裝", DANGER)
            return
        if self.port_in_use():
            self.set_status("已經有一個伺服器在執行（可能是另一個視窗），請先關掉它", DANGER)
            return
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [PYTHON, "-m", "app.server", "--https"], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.ready, self.stage, self.url = False, 0.05, ""
        self.qr_box.pack_forget()
        self.log.delete("1.0", "end")
        self.set_status("啟動中，載入模型約需 30 秒…", ACCENT)
        self.btn.set("停止", "danger")

    def stop(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.ready = False
        self.qr_box.pack_forget()
        self.set_status("已停止", MUTED)
        self.btn.set("啟動", "primary")
        self._draw_bar()

    def on_close(self):
        self.stop()
        self.root.destroy()

    def port_in_use(self):
        """有沒有「視覺助理伺服器」已經在跑。連接埠被別的程式佔用不算：伺服器會自動改用下一個。"""
        import ssl
        import urllib.request
        ctx = ssl._create_unverified_context()
        for p in range(PORT, PORT + 20):
            with socket.socket() as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    continue
            try:
                with urllib.request.urlopen(f"https://127.0.0.1:{p}/ping", timeout=1, context=ctx) as r:
                    if b'"ok"' in r.read():
                        return True
            except Exception:
                pass
        return False

    # ---------------- 讀取伺服器輸出 ----------------
    def _reader(self, proc):
        for raw in proc.stdout:
            self.lines.put(raw.decode("utf-8", "replace").rstrip())
        self.lines.put(None)                              # 程式結束

    def poll(self):
        urls = []
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    if self.proc:                         # 不是按停止，而是自己結束了 → 出錯
                        self.proc = None
                        self.ready = False
                        self.set_status("伺服器意外停止，請看下方紀錄", DANGER)
                        self.btn.set("重新啟動", "primary")
                        self.toggle_log(True)
                    continue
                if "Loading weights" in line:             # 進度條字元太多，不放進紀錄
                    continue
                self.log.insert("end", line + "\n")
                self.log.see("end")
                for key, text, frac in STAGES:
                    if key in line:
                        self.stage = frac
                        self.set_status(text, ACCENT)
                urls += re.findall(r"https?://[^\s）)/]+:\d+", line)   # 只認「主機:連接埠」的伺服器網址，不抓警告裡的連結
                if "伺服器已啟動" in line:
                    self.ready = True
                if "Traceback" in line or "[warn]" in line:
                    self.toggle_log(True)
        except queue.Empty:
            pass
        if urls:
            self.on_urls(urls)
        self.anim += 1
        self._draw_bar()
        self.root.after(100, self.poll)

    def on_urls(self, urls):
        # 優先用 Tailscale 正式網址 → Tailscale IP → 區網 IP
        pick = (next((u for u in urls if ".ts.net" in u), None)
                or next((u for u in urls if "://100." in u), None) or urls[0])
        if self.url and ".ts.net" in self.url:
            return
        self.url = pick
        if not self.ready:
            return
        self.set_status("準備完成！用手機掃描 QR Code", OK)
        import qrcode
        qr = qrcode.QRCode(border=2, box_size=8)
        qr.add_data(self.url)
        qr.make(fit=True)
        im = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize((220, 220), Image.NEAREST)
        self._qr = ImageTk.PhotoImage(im)
        self.qr_label.config(image=self._qr)
        self.url_label.config(text=self.url)
        if ".ts.net" in self.url:
            self.hint.config(text="點網址可以複製・手機需開啟 Tailscale\n想像 App 一樣用：Safari →「分享」→「加入主畫面」")
        else:
            self.hint.config(text="點網址可以複製・手機和電腦要在同一個網路，或開啟 Tailscale\n"
                                  "第一次連線出現憑證警告時，按「顯示詳細資訊」→「造訪此網站」")
        self.qr_box.pack(before=self.btn)

    def copy_url(self):
        if self.url:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.url)
            self.url_label.config(text="已複製！")
            self.root.after(1500, lambda: self.url_label.config(text=self.url))

    def toggle_log(self, show=None):
        show = (not self.log_shown) if show is None else show
        if show and not self.log_shown:
            self.log.pack(pady=(8, 0))
        elif not show and self.log_shown:
            self.log.pack_forget()
        self.log_shown = show
        self.log_toggle.config(text="隱藏紀錄 ▴" if show else "顯示紀錄 ▾")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Launcher().run()
