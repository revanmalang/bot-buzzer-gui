"""
==============================================================================
BOT & BUZZER DETECTION SYSTEM — GUI (Python / CustomTkinter)
Sistem Deteksi Akun Robot & Buzzer Terorganisir berbasis Analisis Perilaku
==============================================================================
Versi antarmuka desktop (GUI) dari engine deteksi bot & buzzer.
Seluruh logika deteksi (frekuensi posting, kesamaan teks, pola waktu) berjalan
murni di Python — tidak butuh browser, tidak butuh API key media sosial.

Cara menjalankan:
    pip install customtkinter
    python3 bot_buzzer_gui.py

Metode Deteksi:
1. Frequency Analysis (Time-Series)   -> kecepatan posting tidak wajar
2. Text Similarity Analysis           -> kemiripan teks antar akun (Jaccard & Cosine)
3. Temporal Pattern Analysis          -> aktivitas 24 jam tanpa jeda istirahat

Fitur GUI:
- Dashboard skor Bot Probability (0-100%) dengan filter kategori
- Ring visual 24-jam per akun (menunjukkan jam-jam akun tersebut aktif)
- Panel bukti (evidence) yang bisa dibuka/tutup per akun
- Ekspor CSV (semua akun / hanya yang terindikasi bot) lewat dialog simpan file
==============================================================================
"""

import csv
import math
import re
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, pstdev
from tkinter import filedialog, messagebox

import customtkinter as ctk

# ==============================================================================
# 1. STRUKTUR DATA
# ==============================================================================

@dataclass
class Post:
    username: str
    timestamp_posting: datetime
    isi_teks: str
    jumlah_posting_per_menit: int


@dataclass
class AccountReport:
    username: str
    total_post: int = 0
    freq_score: float = 0.0
    similarity_score: float = 0.0
    temporal_score: float = 0.0
    bot_probability: float = 0.0
    alasan: list = field(default_factory=list)
    kategori: str = "HUMAN (WAJAR)"
    hours_active: list = field(default_factory=list)
    longest_gap: int = 24


# ==============================================================================
# 2. MOCK DATA
# ==============================================================================

def generate_mock_data():
    posts = []
    base_date = datetime(2026, 8, 24, 0, 0, 0)

    human_texts = {
        "andi_wijaya92": [
            "Baru selesai olahraga pagi, badan enak banget rasanya",
            "Lagi coba resep baru nih, semoga enak",
            "Diskusi menarik banget tadi di kantor soal proyek baru",
            "Capek hari ini, mau istirahat dulu ah",
        ],
        "sitinurhaliza_": [
            "Baca buku baru, ceritanya seru banget",
            "Kucingku lucu banget hari ini hehe",
            "Cuaca mendung, kayaknya mau hujan",
            "Nonton film sama keluarga malam ini",
        ],
        "budi.santoso": [
            "Meeting pagi ini lumayan panjang tapi produktif",
            "Rekomendasi kedai kopi enak di daerah sini dong",
            "Akhirnya kelar juga laporan bulanan",
            "Weekend mau healing kemana ya enaknya",
        ],
    }
    human_hours = [7, 9, 12, 13, 15, 18]

    for username, texts in human_texts.items():
        for i, hour in enumerate(human_hours):
            text = texts[i % len(texts)]
            jitter_minute = hash(username + str(hour)) % 50
            ts = base_date + timedelta(hours=hour, minutes=jitter_minute)
            posts.append(Post(username, ts, text, 1))

    template_narasi = "Ayo dukung program ini sekarang juga sebelum terlambat, jangan sampai menyesal"
    bot_usernames = ["user_88213x", "akun_baru0091", "info_terkini77", "netizen_asli22", "warganet_jujur"]

    coordinated_time = base_date + timedelta(hours=10, minutes=0)
    for idx, username in enumerate(bot_usernames):
        ts = coordinated_time + timedelta(seconds=idx * 15)
        posts.append(Post(username, ts, template_narasi, 14))

    for username in bot_usernames:
        for hour in range(0, 24, 2):
            for burst in range(3):
                ts = base_date + timedelta(hours=hour, minutes=burst * 2)
                posts.append(Post(username, ts, f"{template_narasi} #{burst}", 12 + burst))

    posts.sort(key=lambda p: p.timestamp_posting)
    return posts


# ==============================================================================
# 3. UTILITAS TEKS
# ==============================================================================

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def jaccard_similarity(a, b):
    A, B = set(tokenize(a)), set(tokenize(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine_similarity(a, b):
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    vocab = set(ta) | set(tb)
    va = [ta.count(w) for w in vocab]
    vb = [tb.count(w) for w in vocab]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    return dot / (na * nb) if na and nb else 0.0


def combined_text_similarity(a, b):
    return (jaccard_similarity(a, b) + cosine_similarity(a, b)) / 2


# ==============================================================================
# 4. ENGINE DETEKSI
# ==============================================================================

class BotBuzzerDetector:
    def __init__(self, posts, freq_threshold_per_minute=10, similarity_threshold=0.80,
                 similarity_time_window_minutes=5, rest_gap_threshold_hours=3.0,
                 weight_freq=0.40, weight_similarity=0.35, weight_temporal=0.25):
        self.posts = posts
        self.freq_threshold = freq_threshold_per_minute
        self.similarity_threshold = similarity_threshold
        self.similarity_window = timedelta(minutes=similarity_time_window_minutes)
        self.rest_gap_threshold = timedelta(hours=rest_gap_threshold_hours)
        self.weight_freq = weight_freq
        self.weight_similarity = weight_similarity
        self.weight_temporal = weight_temporal

        self.posts_by_user = defaultdict(list)
        for p in posts:
            self.posts_by_user[p.username].append(p)
        for up in self.posts_by_user.values():
            up.sort(key=lambda p: p.timestamp_posting)

    def analyze_frequency(self, username):
        up = self.posts_by_user[username]
        if len(up) < 2:
            return 0.0, []
        max_reported = max(p.jumlah_posting_per_menit for p in up)
        timestamps = [p.timestamp_posting for p in up]
        max_actual = 1
        for t in timestamps:
            c = sum(1 for t2 in timestamps if 0 <= (t2 - t).total_seconds() <= 60)
            max_actual = max(max_actual, c)
        effective_rate = max(max_reported, max_actual)

        alasan = []
        if effective_rate >= self.freq_threshold:
            ratio = effective_rate / self.freq_threshold
            score = min(100.0, 50 + (ratio - 1) * 50)
            alasan.append(f"Kecepatan posting {effective_rate}/menit melebihi ambang batas wajar "
                          f"({self.freq_threshold}/menit)")
        else:
            score = (effective_rate / self.freq_threshold) * 40
        return round(score, 2), alasan

    def analyze_text_similarity(self):
        result_score, result_cluster = defaultdict(float), defaultdict(set)
        sorted_posts = sorted(self.posts, key=lambda p: p.timestamp_posting)
        for i in range(len(sorted_posts)):
            for j in range(i + 1, len(sorted_posts)):
                p1, p2 = sorted_posts[i], sorted_posts[j]
                if p2.timestamp_posting - p1.timestamp_posting > self.similarity_window:
                    break
                if p1.username == p2.username:
                    continue
                sim = combined_text_similarity(p1.isi_teks, p2.isi_teks)
                if sim >= self.similarity_threshold:
                    score = min(100.0, sim * 100)
                    for u, other in [(p1.username, p2.username), (p2.username, p1.username)]:
                        result_score[u] = max(result_score[u], score)
                        result_cluster[u].add(other)
        result_reason = defaultdict(list)
        for username, partners in result_cluster.items():
            result_reason[username].append(
                f"Menyebarkan teks hampir identik (similarity >= {self.similarity_threshold:.0%}) "
                f"bersama {len(partners)} akun lain: {', '.join(sorted(partners))}")
        return result_score, result_reason

    def analyze_temporal_pattern(self, username):
        up = self.posts_by_user[username]
        if len(up) < 3:
            return 0.0, [], [], 24
        hours_active = sorted(set(p.timestamp_posting.hour for p in up))
        longest_gap = 0
        for i in range(len(hours_active)):
            cur, nxt = hours_active[i], hours_active[(i + 1) % len(hours_active)]
            gap = (nxt - cur) % 24 or 24
            longest_gap = max(longest_gap, gap)

        rest_h = self.rest_gap_threshold.total_seconds() / 3600
        alasan = []
        if longest_gap < rest_h:
            gap_score = min(100.0, (rest_h - longest_gap) / rest_h * 100)
            alasan.append(f"Tidak ditemukan jeda istirahat wajar (celah terpanjang {longest_gap} jam, "
                          f"minimal wajar {rest_h:.0f} jam)")
        else:
            gap_score = 0.0

        hour_counts = defaultdict(int)
        for p in up:
            hour_counts[p.timestamp_posting.hour] += 1
        counts = list(hour_counts.values())
        uniformity_score = 0.0
        if len(counts) >= 8 and mean(counts) > 0:
            cv = pstdev(counts) / mean(counts)
            uniformity_score = max(0.0, (0.5 - cv)) * 100
        if uniformity_score > 20:
            alasan.append("Distribusi jam posting terlalu merata/seragam, tidak menyerupai pola manusia")

        final_score = min(100.0, gap_score * 0.7 + uniformity_score * 0.3)
        return round(final_score, 2), alasan, hours_active, longest_gap

    def run_analysis(self):
        reports = []
        sim_scores, sim_reasons = self.analyze_text_similarity()

        for username in self.posts_by_user:
            freq_score, freq_reason = self.analyze_frequency(username)
            temp_score, temp_reason, hours_active, longest_gap = self.analyze_temporal_pattern(username)
            sim_score = sim_scores.get(username, 0.0)
            sim_reason = sim_reasons.get(username, [])

            bot_prob = round(min(100.0, freq_score * self.weight_freq + sim_score * self.weight_similarity
                                 + temp_score * self.weight_temporal), 2)

            if bot_prob >= 70:
                kategori = "BOT/BUZZER TERINDIKASI KUAT"
            elif bot_prob >= 40:
                kategori = "MENCURIGAKAN"
            else:
                kategori = "HUMAN (WAJAR)"

            alasan = freq_reason + sim_reason + temp_reason
            if not alasan:
                alasan = ["Tidak ditemukan pola anomali signifikan"]

            reports.append(AccountReport(
                username=username, total_post=len(self.posts_by_user[username]),
                freq_score=freq_score, similarity_score=round(sim_score, 2), temporal_score=temp_score,
                bot_probability=bot_prob, alasan=alasan, kategori=kategori,
                hours_active=hours_active, longest_gap=longest_gap,
            ))

        reports.sort(key=lambda r: r.bot_probability, reverse=True)
        return reports


def export_to_csv(reports, filename, only_flagged=False, threshold=70):
    data = [r for r in reports if not only_flagged or r.bot_probability >= threshold]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["username", "bot_probability_score(%)", "kategori", "total_post",
                         "skor_frekuensi", "skor_kesamaan_teks", "skor_pola_waktu", "alasan_deteksi"])
        for r in data:
            writer.writerow([r.username, r.bot_probability, r.kategori, r.total_post,
                             r.freq_score, r.similarity_score, r.temporal_score, " | ".join(r.alasan)])
    return filename, len(data)


# ==============================================================================
# 5. TAMPILAN GUI (CustomTkinter)
# ==============================================================================

COLORS = {
    "ink": "#0B1220", "panel": "#121B2E", "panel2": "#17223A", "hairline": "#223055",
    "text": "#E7ECF5", "muted": "#8592AD",
    "flare": "#FF5A36", "flare_dim": "#33201A",
    "amber": "#F5B940", "amber_dim": "#33290F",
    "teal": "#35D0A6", "teal_dim": "#12312A",
}

F_DISPLAY = ("Space Grotesk", 25, "bold")
F_SUB = ("IBM Plex Sans", 12)
F_MONO_XS = ("IBM Plex Mono", 9)
F_MONO_S = ("IBM Plex Mono", 11)
F_MONO_S_BOLD = ("IBM Plex Mono", 11, "bold")
F_MONO_M_BOLD = ("IBM Plex Mono", 14, "bold")
F_STAT = ("Space Grotesk", 22, "bold")
F_BODY = ("IBM Plex Sans", 11)


def cat_colors(kategori):
    if kategori.startswith("BOT"):
        return COLORS["flare"], COLORS["flare_dim"]
    if kategori.startswith("MENCURIGAKAN"):
        return COLORS["amber"], COLORS["amber_dim"]
    return COLORS["teal"], COLORS["teal_dim"]


class HourRing(tk.Canvas):
    """Ring visual 24-jam — dot menyala menandakan jam akun tersebut aktif posting."""

    def __init__(self, parent, hours_active, color, size=54, **kwargs):
        super().__init__(parent, width=size, height=size, bg=COLORS["panel"],
                          highlightthickness=0, **kwargs)
        cx = cy = size / 2
        R = size / 2 - 7
        active = set(hours_active)
        self.create_oval(cx - R, cy - R, cx + R, cy + R, outline=COLORS["hairline"], width=1)
        for h in range(24):
            angle = (h / 24) * 2 * math.pi - math.pi / 2
            x, y = cx + R * math.cos(angle), cy + R * math.sin(angle)
            r = 3.1 if h in active else 1.3
            fill = color if h in active else "#33405E"
            self.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="")
        self.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=COLORS["muted"], outline="")


class AccountCard(ctk.CTkFrame):
    """Satu baris akun — bisa diklik untuk membuka panel bukti (evidence)."""

    def __init__(self, parent, report, **kwargs):
        super().__init__(parent, fg_color=COLORS["panel"], corner_radius=10, **kwargs)
        self.report = report
        self.open = False
        color, dim = cat_colors(report.kategori)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="x", padx=16, pady=13)
        main.grid_columnconfigure(1, weight=1)

        ring_wrap = ctk.CTkFrame(main, fg_color="transparent")
        ring_wrap.grid(row=0, column=0, padx=(0, 14))
        HourRing(ring_wrap, report.hours_active, color).pack()
        ctk.CTkLabel(ring_wrap, text="24 JAM", font=F_MONO_XS, text_color=COLORS["muted"]).pack(pady=(2, 0))

        info = ctk.CTkFrame(main, fg_color="transparent")
        info.grid(row=0, column=1, sticky="w")
        top = ctk.CTkFrame(info, fg_color="transparent")
        top.pack(anchor="w")
        ctk.CTkLabel(top, text=report.username, font=F_MONO_S_BOLD, text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(top, text=f"  {report.kategori}  ", font=("IBM Plex Mono", 9, "bold"),
                     text_color=color, fg_color=dim, corner_radius=8).pack(side="left", padx=(10, 0))
        ctk.CTkLabel(info, text=f"{report.total_post} postingan dianalisis  ·  celah istirahat terpanjang "
                                 f"{report.longest_gap} jam",
                     font=F_BODY, text_color=COLORS["muted"]).pack(anchor="w", pady=(4, 0))

        score = ctk.CTkFrame(main, fg_color="transparent")
        score.grid(row=0, column=2, sticky="e", padx=(10, 0))
        bar = ctk.CTkProgressBar(score, width=120, height=7, progress_color=color,
                                  fg_color=COLORS["hairline"], corner_radius=4)
        bar.set(report.bot_probability / 100)
        bar.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(score, text=f"{report.bot_probability}%", font=F_MONO_M_BOLD,
                     text_color=color, width=64, anchor="e").pack(side="left")
        self.chevron = ctk.CTkLabel(score, text="\u25be", font=F_MONO_S, text_color=COLORS["muted"])
        self.chevron.pack(side="left", padx=(4, 0))

        self.detail = ctk.CTkFrame(self, fg_color=COLORS["panel2"], corner_radius=8)
        self._build_detail(color)

        for w in (self, main, info, top, score):
            w.bind("<Button-1>", self.toggle)

    def _build_detail(self, color):
        subs = [("FREKUENSI", self.report.freq_score), ("KESAMAAN TEKS", self.report.similarity_score),
                ("POLA WAKTU", self.report.temporal_score)]
        row = ctk.CTkFrame(self.detail, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(14, 6))
        for label, val in subs:
            block = ctk.CTkFrame(row, fg_color="transparent")
            block.pack(side="left", expand=True, fill="x", padx=(0, 18))
            head = ctk.CTkFrame(block, fg_color="transparent")
            head.pack(fill="x")
            ctk.CTkLabel(head, text=label, font=F_MONO_XS, text_color=COLORS["muted"]).pack(side="left")
            ctk.CTkLabel(head, text=str(val), font=("IBM Plex Mono", 10, "bold"),
                         text_color=COLORS["text"]).pack(side="right")
            mini = ctk.CTkProgressBar(block, height=4, progress_color=COLORS["muted"],
                                      fg_color=COLORS["hairline"], corner_radius=2)
            mini.set(val / 100)
            mini.pack(fill="x", pady=(4, 0))

        for a in self.report.alasan:
            ctk.CTkLabel(self.detail, text=f"\u2022  {a}", font=F_BODY, text_color=COLORS["text"],
                         wraplength=860, justify="left", anchor="w").pack(fill="x", padx=22, pady=2)
        ctk.CTkFrame(self.detail, fg_color="transparent", height=8).pack()

    def toggle(self, event=None):
        self.open = not self.open
        if self.open:
            self.detail.pack(fill="x", padx=14, pady=(0, 14))
            self.chevron.configure(text="\u25b4")
        else:
            self.detail.pack_forget()
            self.chevron.configure(text="\u25be")


class BotBuzzerApp(ctk.CTk):
    def __init__(self, reports):
        super().__init__()
        self.all_reports = reports
        self.title("Bot & Buzzer Detection System")
        self.geometry("1060x780")
        self.configure(fg_color=COLORS["ink"])

        self._build_header()
        self._build_stats()
        self._build_controls()

        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent",
                                                  scrollbar_button_color=COLORS["hairline"])
        self.list_frame.pack(fill="both", expand=True, padx=26, pady=(0, 8))

        self._build_footer()
        self.render_cards(self.all_reports)

    def _build_header(self):
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(fill="x", padx=26, pady=(22, 8))

        top = ctk.CTkFrame(wrap, fg_color="transparent")
        top.pack(fill="x")
        left = ctk.CTkFrame(top, fg_color="transparent")
        left.pack(side="left", anchor="n")
        ctk.CTkLabel(left, text="\u25cf SOCIAL CYBER DEFENSE // BOT & BUZZER DETECTION SYSTEM",
                     font=F_MONO_XS, text_color=COLORS["teal"]).pack(anchor="w")
        ctk.CTkLabel(left, text="Deteksi Akun Bot & Buzzer", font=F_DISPLAY,
                     text_color=COLORS["text"]).pack(anchor="w", pady=(4, 2))
        ctk.CTkLabel(left, text="Analisis perilaku akun murni berbasis data — frekuensi posting,\n"
                                "kesamaan teks antar akun, dan pola jam aktif.",
                     font=F_SUB, text_color=COLORS["muted"], justify="left").pack(anchor="w")

        right = ctk.CTkFrame(top, fg_color="transparent")
        right.pack(side="right", anchor="n")
        for line in ["MODE: HEURISTIK & STATISTIK", "DATASET: MOCK DATA (DEMO)", "AMBANG BOT: SKOR \u2265 70%"]:
            ctk.CTkLabel(right, text=line, font=F_MONO_XS, text_color=COLORS["muted"],
                         anchor="e").pack(anchor="e", pady=1)

        ctk.CTkFrame(self, height=1, fg_color=COLORS["hairline"]).pack(fill="x", padx=26, pady=(14, 0))

    def _build_stats(self):
        strip = ctk.CTkFrame(self, fg_color="transparent")
        strip.pack(fill="x", padx=26, pady=16)
        total = len(self.all_reports)
        bot = len([r for r in self.all_reports if r.kategori.startswith("BOT")])
        susp = len([r for r in self.all_reports if r.kategori.startswith("MENCURIGAKAN")])
        avg = sum(r.bot_probability for r in self.all_reports) / total if total else 0

        tiles = [
            ("TOTAL AKUN DISCAN", str(total), COLORS["hairline"], COLORS["text"]),
            ("TERINDIKASI BOT", str(bot), COLORS["flare"], COLORS["flare"]),
            ("MENCURIGAKAN", str(susp), COLORS["amber"], COLORS["amber"]),
            ("RATA-RATA SKOR", f"{avg:.1f}%", COLORS["teal"], COLORS["teal"]),
        ]
        for label, value, border, color in tiles:
            tile = ctk.CTkFrame(strip, fg_color=COLORS["panel"], corner_radius=10)
            tile.pack(side="left", expand=True, fill="both", padx=6)
            ctk.CTkFrame(tile, height=3, fg_color=border, corner_radius=0).pack(fill="x")
            ctk.CTkLabel(tile, text=label, font=F_MONO_XS, text_color=COLORS["muted"],
                         anchor="w").pack(fill="x", padx=16, pady=(12, 2))
            ctk.CTkLabel(tile, text=value, font=F_STAT, text_color=color,
                         anchor="w").pack(fill="x", padx=16, pady=(0, 14))

    def _build_controls(self):
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.pack(fill="x", padx=26, pady=(0, 10))
        self.filter_var = tk.StringVar(value="Semua Akun")
        seg = ctk.CTkSegmentedButton(
            ctrl, values=["Semua Akun", "Terindikasi Bot", "Mencurigakan", "Manusia"],
            variable=self.filter_var, command=self.on_filter, font=F_MONO_XS,
            selected_color=COLORS["teal"], selected_hover_color=COLORS["teal"],
            unselected_color=COLORS["panel"], fg_color=COLORS["panel"],
            text_color=COLORS["text"], text_color_disabled=COLORS["muted"],
        )
        seg.pack(anchor="w")

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=26, pady=(4, 18))
        ctk.CTkLabel(footer, text="Data pada dashboard ini adalah data tiruan (mock) untuk pengujian sistem\n"
                                  "— bukan data akun nyata dari platform manapun.",
                     font=("IBM Plex Sans", 9), text_color=COLORS["muted"],
                     justify="left").pack(side="left")
        btns = ctk.CTkFrame(footer, fg_color="transparent")
        btns.pack(side="right")
        ctk.CTkButton(btns, text="\u2b73  EKSPOR SEMUA (CSV)", font=F_MONO_XS,
                     fg_color=COLORS["panel"], hover_color=COLORS["panel2"], text_color=COLORS["text"],
                     border_width=1, border_color=COLORS["hairline"],
                     command=lambda: self.do_export(False)).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="\u2b73  EKSPOR TERINDIKASI BOT (CSV)", font=F_MONO_XS,
                     fg_color=COLORS["flare_dim"], hover_color=COLORS["flare"], text_color=COLORS["flare"],
                     command=lambda: self.do_export(True)).pack(side="left", padx=6)

    def on_filter(self, value):
        mapping = {"Semua Akun": None, "Terindikasi Bot": "BOT",
                  "Mencurigakan": "MENCURIGAKAN", "Manusia": "HUMAN"}
        key = mapping[value]
        data = self.all_reports if key is None else [r for r in self.all_reports if r.kategori.startswith(key)]
        self.render_cards(data)

    def render_cards(self, reports):
        for w in self.list_frame.winfo_children():
            w.destroy()
        for r in reports:
            card = AccountCard(self.list_frame, r)
            card.pack(fill="x", pady=5)

    def do_export(self, only_flagged):
        default_name = "bot_detection_flagged_for_blocklist.csv" if only_flagged else "bot_detection_full_report.csv"
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default_name,
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        filename, count = export_to_csv(self.all_reports, path, only_flagged=only_flagged)
        messagebox.showinfo("Ekspor selesai", f"{count} akun berhasil diekspor ke:\n{filename}")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    ctk.set_appearance_mode("dark")
    posts = generate_mock_data()
    detector = BotBuzzerDetector(posts)
    reports = detector.run_analysis()
    app = BotBuzzerApp(reports)
    app.mainloop()


if __name__ == "__main__":
    main()
