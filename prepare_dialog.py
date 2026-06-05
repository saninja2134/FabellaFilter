# Modal for configuring dataset generation and augmentation, inspired by Roboflow's UI.
import tkinter as tk
from tkinter import ttk
import json

BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
SECTION_BG = "#252526"
INPUT_BG = "#3C3C3C"
ACCENT_COLOR = "#007ACC"
ACCENT_GREEN = "#4CAF50"
BUTTON_BG = "#333333"
BORDER_COLOR = "#3E3E3E"
ACCENT_RED = "#F44747"


class RangeSlider(tk.Canvas):
    # A dual-handle range slider widget drawn on a Canvas.
    def __init__(self, parent, min_val, max_val, init_low, init_high, unit="", step=1, width=300, **kwargs):
        super().__init__(parent, height=50, bg=SECTION_BG, highlightthickness=0, width=width, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.low = float(init_low)
        self.high = float(init_high)
        self.unit = unit
        self.step = float(step)
        self.dragging = None
        self.TRACK_Y = 22
        self.THUMB_R = 7
        self.PAD = 12
        self._callbacks = []
        self.bind('<Button-1>', self._on_press)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Configure>', lambda e: self.after(10, self._redraw))
        self.after(50, self._redraw)

    def _val_to_x(self, val):
        w = self.winfo_width() or 300
        usable = w - 2 * self.PAD
        frac = (val - self.min_val) / (self.max_val - self.min_val) if self.max_val != self.min_val else 0
        return self.PAD + frac * usable

    def _x_to_val(self, x):
        w = self.winfo_width() or 300
        usable = w - 2 * self.PAD
        frac = max(0.0, min(1.0, (x - self.PAD) / usable))
        raw = self.min_val + frac * (self.max_val - self.min_val)
        snapped = round(raw / self.step) * self.step
        return max(self.min_val, min(self.max_val, snapped))

    def _redraw(self):
        self.delete('all')
        w = self.winfo_width() or 300
        y = self.TRACK_Y
        r = self.THUMB_R

        lx = self._val_to_x(self.low)
        hx = self._val_to_x(self.high)

        # Background track
        self.create_rectangle(self.PAD, y - 3, w - self.PAD, y + 3,
                              fill='#444444', outline='', tags='track_bg')
        # Active range
        self.create_rectangle(lx, y - 3, hx, y + 3,
                              fill=ACCENT_COLOR, outline='', tags='track_active')
        # Low thumb
        self.create_oval(lx - r, y - r, lx + r, y + r,
                        fill='#FFFFFF', outline=ACCENT_COLOR, width=2, tags='low_thumb')
        # High thumb
        self.create_oval(hx - r, y - r, hx + r, y + r,
                        fill='#FFFFFF', outline=ACCENT_COLOR, width=2, tags='high_thumb')

        # Value labels
        low_str = f"{int(self.low)}{self.unit}" if self.step >= 1 else f"{self.low:.1f}{self.unit}"
        high_str = f"{int(self.high)}{self.unit}" if self.step >= 1 else f"{self.high:.1f}{self.unit}"
        self.create_text(lx, y + r + 10, text=low_str, fill='#AAAAAA', font=('Segoe UI', 8), tags='label_low')
        self.create_text(hx, y + r + 10, text=high_str, fill='#AAAAAA', font=('Segoe UI', 8), tags='label_high')

    def _on_press(self, event):
        lx = self._val_to_x(self.low)
        hx = self._val_to_x(self.high)
        y = self.TRACK_Y
        r = self.THUMB_R + 6
        dl = ((event.x - lx) ** 2 + (event.y - y) ** 2) ** 0.5
        dh = ((event.x - hx) ** 2 + (event.y - y) ** 2) ** 0.5
        if dl < r and dl <= dh:
            self.dragging = 'low'
        elif dh < r:
            self.dragging = 'high'

    def _on_drag(self, event):
        if not self.dragging:
            return
        val = self._x_to_val(event.x)
        if self.dragging == 'low':
            self.low = min(val, self.high)
        else:
            self.high = max(val, self.low)
        self._redraw()
        for cb in self._callbacks:
            cb(self.low, self.high)

    def _on_release(self, event):
        self.dragging = None

    def get(self):
        return (self.low, self.high)

    def set(self, low, high):
        self.low = float(low)
        self.high = float(high)
        self._redraw()

    def trace(self, callback):
        self._callbacks.append(callback)


class ToggleSwitch(tk.Canvas):
    # A simple iOS-style toggle switch.
    def __init__(self, parent, initial=False, command=None, **kwargs):
        super().__init__(parent, width=40, height=22, bg=SECTION_BG,
                        highlightthickness=0, cursor='hand2', **kwargs)
        self.state = initial
        self.command = command
        self.bind('<Button-1>', self._toggle)
        self._redraw()

    def _redraw(self):
        self.delete('all')
        color = ACCENT_COLOR if self.state else '#555555'
        self.create_rectangle(0, 2, 40, 20, fill=color, outline='', tags='bg')
        self.create_oval(0, 0, 40, 22, fill=color, outline='')
        # Simpler rounded rect via oval at ends
        cx = 28 if self.state else 12
        self.create_oval(cx - 8, 3, cx + 8, 19, fill='white', outline='')

    def _toggle(self, event=None):
        self.state = not self.state
        self._redraw()
        if self.command:
            self.command(self.state)

    def get(self):
        return self.state

    def set(self, val):
        self.state = bool(val)
        self._redraw()


class ScrollableFrame(tk.Frame):
    # A vertically scrollable frame container.
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=BG_COLOR)

        self.inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind scroll only while the mouse is inside this frame to avoid
        # firing on destroyed widgets after the modal closes.
        def _on_scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')

        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_scroll))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))
        self.inner.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_scroll))
        self.inner.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))


class SectionCard(tk.Frame):
    # A styled card frame for a settings section.
    def __init__(self, parent, title, badge_text="", **kwargs):
        super().__init__(parent, bg=SECTION_BG, highlightbackground=BORDER_COLOR,
                        highlightthickness=1, **kwargs)
        self.pack_propagate(True)

        header = tk.Frame(self, bg=SECTION_BG, pady=12, padx=15)
        header.pack(fill=tk.X)

        badge_color = ACCENT_COLOR
        if badge_text == "AUGMENTATION":
            badge_color = "#7B2FBE"
        elif badge_text == "GENERATION":
            badge_color = "#2E7D32"

        if badge_text:
            badge = tk.Label(header, text=badge_text, font=('Segoe UI', 7, 'bold'),
                            bg=badge_color, fg='white', padx=6, pady=2)
            badge.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(header, text=title, font=('Segoe UI', 11, 'bold'),
                bg=SECTION_BG, fg=FG_COLOR).pack(side=tk.LEFT)

        sep = tk.Frame(self, bg=BORDER_COLOR, height=1)
        sep.pack(fill=tk.X, padx=0)

        self.body = tk.Frame(self, bg=SECTION_BG, padx=15, pady=10)
        self.body.pack(fill=tk.X)


def label(parent, text, small=False):
    size = 9 if small else 10
    return tk.Label(parent, text=text, font=('Segoe UI', size), bg=SECTION_BG, fg=FG_COLOR)


def sublabel(parent, text):
    return tk.Label(parent, text=text, font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888')


class AugRow(tk.Frame):
    # A row representing one augmentation option that can be expanded.
    def __init__(self, parent, aug_name, build_controls_fn, **kwargs):
        super().__init__(parent, bg=SECTION_BG, **kwargs)
        self.aug_name = aug_name
        self.enabled_var = tk.BooleanVar(value=False)
        self.controls_frame = None

        header = tk.Frame(self, bg="#2A2A2A", padx=10, pady=8)
        header.pack(fill=tk.X, pady=(2, 0))
        header.columnconfigure(1, weight=1)
        header.grid_propagate(True)

        # Toggle
        self.toggle = ToggleSwitch(header, initial=False, command=self._on_toggle)
        self.toggle.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header, text=aug_name, font=('Segoe UI', 10, 'bold'),
                bg="#2A2A2A", fg=FG_COLOR).pack(side=tk.LEFT)

        # Collapsible controls area
        self.expand_frame = tk.Frame(self, bg="#222222", padx=15, pady=10)
        self.build_controls = build_controls_fn
        self._controls_built = False

    def _on_toggle(self, state):
        self.enabled_var.set(state)
        if state:
            if not self._controls_built:
                self.build_controls(self.expand_frame)
                self._controls_built = True
            self.expand_frame.pack(fill=tk.X)
        else:
            self.expand_frame.pack_forget()

    def get_enabled(self):
        return self.enabled_var.get()


class DatasetGeneratorModal(tk.Toplevel):
    # A Roboflow-inspired modal for configuring dataset generation and augmentation.
    def __init__(self, parent, task="segment", pos_sorted_dir="data/sorted/pos", on_generate=None):
        super().__init__(parent)
        self.title("Dataset Generation & Augmentation")
        self.geometry("780x880")
        self.resizable(True, True)
        self.configure(bg=BG_COLOR)
        self.on_generate = on_generate
        self.task = task
        self.pos_sorted_dir = pos_sorted_dir

        # Store control references
        self.ctrl = {}

        # Store augmentation row references
        self.aug_rows = {}

        self._build_ui()
        self.transient(parent)
        self.grab_set()

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg="#007ACC", padx=20, pady=14)
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="Dataset Generation & Augmentation",
                font=('Segoe UI', 13, 'bold'), bg="#007ACC", fg='white').pack(side=tk.LEFT)
        task_badge = "SEG" if self.task == "segment" else "OBB" if self.task == "obb" else "DET"
        tk.Label(title_bar, text=f"{task_badge} + COCO + DET", font=('Segoe UI', 9, 'bold'),
                bg='#005A9E', fg='white', padx=8, pady=3).pack(side=tk.RIGHT)

        # Main scroll area
        scroll = ScrollableFrame(self, bg=BG_COLOR)
        scroll.pack(fill=tk.BOTH, expand=True)
        content = scroll.inner

        # === SECTION 1: PREPROCESSING ===
        pre_card = SectionCard(content, "Preprocessing", badge_text="PREPROCESSING")
        pre_card.pack(fill=tk.X, padx=15, pady=(15, 8))

        sublabel(pre_card.body, "Applied once to all images before augmentation.").pack(anchor='w', pady=(0, 12))

        self._build_preprocessing(pre_card.body)

        # === SECTION 2: AUGMENTATIONS ===
        aug_card = SectionCard(content, "Image-Level Augmentations", badge_text="AUGMENTATION")
        aug_card.pack(fill=tk.X, padx=15, pady=8)

        sublabel(aug_card.body, "Applied randomly to generated copies. Enable each augmentation to configure it.").pack(anchor='w', pady=(0, 12))

        self._build_augmentations(aug_card.body)

        # === SECTION 3: GENERATION SETTINGS ===
        gen_card = SectionCard(content, "Generation Settings", badge_text="GENERATION")
        gen_card.pack(fill=tk.X, padx=15, pady=(8, 80))

        self._build_generation(gen_card.body)

        # Footer (sticky)
        footer = tk.Frame(self, bg="#1A1A1A", padx=20, pady=14,
                         highlightbackground=BORDER_COLOR, highlightthickness=1)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(footer, text="Cancel", font=('Segoe UI', 10),
                 bg=BUTTON_BG, fg=FG_COLOR, activebackground="#444444",
                 relief=tk.FLAT, padx=20, cursor='hand2',
                 command=self.destroy).pack(side=tk.LEFT)

        self.gen_btn = tk.Button(footer, text="⚡  Generate Dataset (All Formats)",
                                font=('Segoe UI', 11, 'bold'),
                                bg=ACCENT_GREEN, fg='white',
                                activebackground='#388E3C',
                                relief=tk.FLAT, padx=24, pady=6,
                                cursor='hand2', command=self._on_generate)
        self.gen_btn.pack(side=tk.RIGHT)

        self.count_label = tk.Label(footer, text="~0 images will be generated",
                                   font=('Segoe UI', 9), bg="#1A1A1A", fg='#AAAAAA')
        self.count_label.pack(side=tk.RIGHT, padx=20)

    # ────────────────────────────────────────────────────────────
    # SECTION 1 BUILDERS
    # ────────────────────────────────────────────────────────────

    def _build_preprocessing(self, parent):
        def row(f, left_widget, label_text, desc=None):
            r = tk.Frame(f, bg=SECTION_BG)
            r.pack(fill=tk.X, pady=4)
            left_widget_instance = left_widget(r)
            left_widget_instance.pack(side=tk.LEFT)
            col = tk.Frame(r, bg=SECTION_BG)
            col.pack(side=tk.LEFT, padx=12)
            tk.Label(col, text=label_text, font=('Segoe UI', 10, 'bold'),
                    bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
            if desc:
                tk.Label(col, text=desc, font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(anchor='w')
            return left_widget_instance

        # Auto-Orient Toggle
        orient_var = ToggleSwitch(None)
        r = tk.Frame(parent, bg=SECTION_BG); r.pack(fill=tk.X, pady=4)
        t = ToggleSwitch(r, initial=True)
        t.pack(side=tk.LEFT)
        col = tk.Frame(r, bg=SECTION_BG); col.pack(side=tk.LEFT, padx=12)
        tk.Label(col, text="Auto-Orient", font=('Segoe UI', 10, 'bold'), bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
        tk.Label(col, text="Discard EXIF data; standardize image rotation.", font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(anchor='w')
        self.ctrl['auto_orient'] = t

        # Resize
        r2 = tk.Frame(parent, bg=SECTION_BG); r2.pack(fill=tk.X, pady=8)
        tk.Label(r2, text="Resize", font=('Segoe UI', 10, 'bold'), bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
        resize_row = tk.Frame(parent, bg=SECTION_BG); resize_row.pack(fill=tk.X, pady=(0, 4))

        mode_var = tk.StringVar(value="Stretch")
        mode_cb = ttk.Combobox(resize_row, textvariable=mode_var,
                               values=["Stretch", "Fit (White Pad)", "Fit (Black Pad)", "Crop"],
                               state='readonly', width=16)
        mode_cb.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(resize_row, text="W:", font=('Segoe UI', 9), bg=SECTION_BG, fg='#AAAAAA').pack(side=tk.LEFT)
        w_var = tk.IntVar(value=1024)
        tk.Spinbox(resize_row, textvariable=w_var, from_=32, to=4096, width=6,
                  bg=INPUT_BG, fg=FG_COLOR, buttonbackground=INPUT_BG,
                  insertbackground=FG_COLOR, relief=tk.FLAT).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(resize_row, text="H:", font=('Segoe UI', 9), bg=SECTION_BG, fg='#AAAAAA').pack(side=tk.LEFT)
        h_var = tk.IntVar(value=1024)
        tk.Spinbox(resize_row, textvariable=h_var, from_=32, to=4096, width=6,
                  bg=INPUT_BG, fg=FG_COLOR, buttonbackground=INPUT_BG,
                  insertbackground=FG_COLOR, relief=tk.FLAT).pack(side=tk.LEFT, padx=(2, 0))
        tk.Label(resize_row, text="px each", font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(side=tk.LEFT, padx=4)
        self.ctrl['resize_mode'] = mode_var
        self.ctrl['resize_w'] = w_var
        self.ctrl['resize_h'] = h_var

        # Auto-Contrast Toggle
        r3 = tk.Frame(parent, bg=SECTION_BG); r3.pack(fill=tk.X, pady=4)
        t3 = ToggleSwitch(r3, initial=False)
        t3.pack(side=tk.LEFT)
        col3 = tk.Frame(r3, bg=SECTION_BG); col3.pack(side=tk.LEFT, padx=12)
        tk.Label(col3, text="Auto-Adjust Contrast", font=('Segoe UI', 10, 'bold'), bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
        tk.Label(col3, text="Apply CLAHE histogram equalization.", font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(anchor='w')
        self.ctrl['auto_contrast'] = t3

        # Grayscale Toggle
        r4 = tk.Frame(parent, bg=SECTION_BG); r4.pack(fill=tk.X, pady=4)
        t4 = ToggleSwitch(r4, initial=False)
        t4.pack(side=tk.LEFT)
        col4 = tk.Frame(r4, bg=SECTION_BG); col4.pack(side=tk.LEFT, padx=12)
        tk.Label(col4, text="Grayscale", font=('Segoe UI', 10, 'bold'), bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
        tk.Label(col4, text="Convert images to single-channel grayscale.", font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(anchor='w')
        self.ctrl['grayscale'] = t4

    # ────────────────────────────────────────────────────────────
    # SECTION 2 BUILDERS
    # ────────────────────────────────────────────────────────────

    def _build_augmentations(self, parent):
        aug_defs = [
            ("Flip",          self._build_flip_controls),
            ("90° Rotate",    self._build_rotate90_controls),
            ("Rotation",      self._build_rotation_controls),
            ("Crop",          self._build_crop_controls),
            ("Shear",         self._build_shear_controls),
            ("Brightness",    self._build_brightness_controls),
            ("Exposure",      self._build_exposure_controls),
            ("Saturation",    self._build_saturation_controls),
            ("Hue",           self._build_hue_controls),
            ("Blur",          self._build_blur_controls),
            ("Noise",         self._build_noise_controls),
            ("Mosaic",        self._build_mosaic_controls),
        ]
        for name, builder in aug_defs:
            row = AugRow(parent, name, builder)
            row.pack(fill=tk.X, pady=2)
            self.aug_rows[name] = row

    def _range_row(self, parent, label_text, min_val, max_val, init_low, init_high, unit="", step=1):
        f = tk.Frame(parent, bg="#222222"); f.pack(fill=tk.X, pady=4)
        tk.Label(f, text=label_text, font=('Segoe UI', 9), bg="#222222", fg='#AAAAAA', width=20, anchor='w').pack(side=tk.LEFT)
        slider = RangeSlider(f, min_val, max_val, init_low, init_high, unit=unit, step=step)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        return slider

    def _single_slider_row(self, parent, label_text, min_val, max_val, init_val, unit=""):
        f = tk.Frame(parent, bg="#222222"); f.pack(fill=tk.X, pady=4)
        tk.Label(f, text=label_text, font=('Segoe UI', 9), bg="#222222", fg='#AAAAAA', width=20, anchor='w').pack(side=tk.LEFT)
        var = tk.DoubleVar(value=init_val)
        val_label = tk.Label(f, text=f"{init_val:.0f}{unit}", font=('Segoe UI', 9), bg="#222222", fg=ACCENT_COLOR, width=6)
        val_label.pack(side=tk.RIGHT)
        scale = tk.Scale(f, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                        variable=var, bg="#222222", fg=FG_COLOR,
                        troughcolor='#444444', highlightthickness=0, showvalue=0,
                        command=lambda v: val_label.config(text=f"{float(v):.0f}{unit}"))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        return var

    def _checkbox_row(self, parent, options):
        f = tk.Frame(parent, bg="#222222"); f.pack(fill=tk.X, pady=4)
        vars_ = {}
        for opt in options:
            v = tk.BooleanVar(value=True)
            tk.Checkbutton(f, text=opt, variable=v, bg="#222222", fg=FG_COLOR,
                          selectcolor="#444444", activebackground="#222222",
                          font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 15))
            vars_[opt] = v
        return vars_

    def _build_flip_controls(self, parent):
        vars_ = self._checkbox_row(parent, ["Horizontal", "Vertical"])
        self.ctrl['flip'] = vars_

    def _build_rotate90_controls(self, parent):
        vars_ = self._checkbox_row(parent, ["Clockwise", "Counter-Clockwise", "Upside Down"])
        self.ctrl['rotate90'] = vars_

    def _build_rotation_controls(self, parent):
        s = self._range_row(parent, "Rotation range", -180, 180, -45, 45, unit="°")
        self.ctrl['rotation'] = s

    def _build_crop_controls(self, parent):
        s = self._range_row(parent, "Zoom %", 0, 100, 0, 50, unit="%")
        self.ctrl['crop'] = s

    def _build_shear_controls(self, parent):
        sh = self._single_slider_row(parent, "Horizontal shear", -45, 45, 15, unit="°")
        sv = self._single_slider_row(parent, "Vertical shear", -45, 45, 15, unit="°")
        self.ctrl['shear_h'] = sh
        self.ctrl['shear_v'] = sv

    def _build_brightness_controls(self, parent):
        s = self._range_row(parent, "Brightness", -100, 100, -25, 25, unit="%")
        self.ctrl['brightness'] = s

    def _build_exposure_controls(self, parent):
        s = self._range_row(parent, "Exposure", -100, 100, -15, 15, unit="%")
        self.ctrl['exposure'] = s

    def _build_saturation_controls(self, parent):
        s = self._range_row(parent, "Saturation", -100, 100, -25, 25, unit="%")
        self.ctrl['saturation'] = s

    def _build_hue_controls(self, parent):
        s = self._range_row(parent, "Hue shift", -180, 180, -15, 15, unit="°")
        self.ctrl['hue'] = s

    def _build_blur_controls(self, parent):
        v = self._single_slider_row(parent, "Max blur", 1, 10, 3, unit="px")
        self.ctrl['blur'] = v

    def _build_noise_controls(self, parent):
        v = self._single_slider_row(parent, "Max noise", 0, 15, 5, unit="%")
        self.ctrl['noise'] = v

    def _build_mosaic_controls(self, parent):
        tk.Label(parent, text="Combine 4 random images into one YOLO mosaic tile.",
                font=('Segoe UI', 9), bg="#222222", fg='#888888').pack(anchor='w', pady=4)

    # ────────────────────────────────────────────────────────────
    # SECTION 3 BUILDERS
    # ────────────────────────────────────────────────────────────

    def _build_generation(self, parent):
        row = tk.Frame(parent, bg=SECTION_BG); row.pack(fill=tk.X, pady=4)
        tk.Label(row, text="Output Multiplier:", font=('Segoe UI', 10, 'bold'),
                bg=SECTION_BG, fg=FG_COLOR).pack(side=tk.LEFT)

        self.multiplier_var = tk.StringVar(value="3x")
        mult_values = ["1x (No Augmentation)", "2x", "3x", "5x", "10x"]
        mult_cb = ttk.Combobox(row, textvariable=self.multiplier_var,
                               values=mult_values, state='readonly', width=20)
        mult_cb.pack(side=tk.LEFT, padx=15)
        mult_cb.bind('<<ComboboxSelected>>', self._update_count_label)

        self.count_preview = tk.Label(parent, text="", font=('Segoe UI', 10),
                                     bg=SECTION_BG, fg='#AAAAAA')
        self.count_preview.pack(anchor='w', pady=(6, 0))
        self._update_count_label()

        # Preserve Originals toggle
        sep = tk.Frame(parent, bg=BORDER_COLOR, height=1)
        sep.pack(fill=tk.X, pady=(12, 8))
        preserve_row = tk.Frame(parent, bg=SECTION_BG)
        preserve_row.pack(fill=tk.X, pady=4)
        preserve_toggle = ToggleSwitch(preserve_row, initial=False)
        preserve_toggle.pack(side=tk.LEFT)
        col = tk.Frame(preserve_row, bg=SECTION_BG)
        col.pack(side=tk.LEFT, padx=12)
        tk.Label(col, text="Preserve Originals", font=('Segoe UI', 10, 'bold'),
                bg=SECTION_BG, fg=FG_COLOR).pack(anchor='w')
        tk.Label(col, text="Include one raw unprocessed copy of every source image alongside augmented versions.",
                font=('Segoe UI', 9), bg=SECTION_BG, fg='#888888').pack(anchor='w')
        self.ctrl['preserve_originals'] = preserve_toggle

    def _update_count_label(self, event=None):
        val = self.multiplier_var.get()
        try:
            mult = int(val.replace('x', '').split(' ')[0])
            # Estimate based on what's in the sorted folder
            import os
            src = self.pos_sorted_dir
            n_src = len([f for f in os.listdir(src) if f.endswith('.png')]) if os.path.exists(src) else 0
            estimated = n_src * mult
            self.count_preview.config(text=f"Generating roughly {estimated:,} images  ({n_src:,} source × {mult}x multiplier)")
            if hasattr(self, 'count_label'):
                self.count_label.config(text=f"~{estimated:,} images")
        except Exception:
            if hasattr(self, 'count_preview'):
                self.count_preview.config(text="")

    # ────────────────────────────────────────────────────────────
    # CONFIG EXTRACTION
    # ────────────────────────────────────────────────────────────

    def _collect_config(self):
        def get_slider(key):
            ctrl = self.ctrl.get(key)
            if ctrl is None: return None
            if isinstance(ctrl, RangeSlider):
                return list(ctrl.get())
            elif isinstance(ctrl, tk.DoubleVar):
                return ctrl.get()
            return None

        def aug_enabled(name):
            return self.aug_rows[name].get_enabled()

        mult_str = self.multiplier_var.get()
        multiplier = int(mult_str.replace('x', '').split(' ')[0])

        config = {
            "preprocessing": {
                "auto_orient": self.ctrl['auto_orient'].get(),
                "resize": {
                    "mode":   self.ctrl['resize_mode'].get(),
                    "width":  self.ctrl['resize_w'].get(),
                    "height": self.ctrl['resize_h'].get(),
                },
                "auto_contrast": self.ctrl['auto_contrast'].get(),
                "grayscale":     self.ctrl['grayscale'].get(),
            },
            "augmentations": {
                "flip": {
                    "enabled":    aug_enabled("Flip"),
                    "horizontal": self.ctrl.get('flip', {}).get('Horizontal', tk.BooleanVar(value=True)).get() if aug_enabled("Flip") else False,
                    "vertical":   self.ctrl.get('flip', {}).get('Vertical', tk.BooleanVar(value=False)).get() if aug_enabled("Flip") else False,
                },
                "rotate90": {
                    "enabled":          aug_enabled("90° Rotate"),
                    "clockwise":        self.ctrl.get('rotate90', {}).get('Clockwise', tk.BooleanVar(value=True)).get() if aug_enabled("90° Rotate") else False,
                    "counter_clockwise": self.ctrl.get('rotate90', {}).get('Counter-Clockwise', tk.BooleanVar(value=True)).get() if aug_enabled("90° Rotate") else False,
                    "upside_down":      self.ctrl.get('rotate90', {}).get('Upside Down', tk.BooleanVar(value=True)).get() if aug_enabled("90° Rotate") else False,
                },
                "rotation": {
                    "enabled": aug_enabled("Rotation"),
                    "range":   get_slider('rotation') or [-45, 45],
                },
                "crop": {
                    "enabled": aug_enabled("Crop"),
                    "range":   get_slider('crop') or [0, 50],
                },
                "shear": {
                    "enabled":    aug_enabled("Shear"),
                    "horizontal": self.ctrl.get('shear_h', tk.DoubleVar(value=15)).get(),
                    "vertical":   self.ctrl.get('shear_v', tk.DoubleVar(value=15)).get(),
                },
                "brightness": {
                    "enabled": aug_enabled("Brightness"),
                    "range":   get_slider('brightness') or [-25, 25],
                },
                "exposure": {
                    "enabled": aug_enabled("Exposure"),
                    "range":   get_slider('exposure') or [-15, 15],
                },
                "saturation": {
                    "enabled": aug_enabled("Saturation"),
                    "range":   get_slider('saturation') or [-25, 25],
                },
                "hue": {
                    "enabled": aug_enabled("Hue"),
                    "range":   get_slider('hue') or [-15, 15],
                },
                "blur": {
                    "enabled": aug_enabled("Blur"),
                    "max_px":  self.ctrl.get('blur', tk.DoubleVar(value=3)).get(),
                },
                "noise": {
                    "enabled": aug_enabled("Noise"),
                    "max_pct": self.ctrl.get('noise', tk.DoubleVar(value=5)).get(),
                },
                "mosaic": {
                    "enabled": aug_enabled("Mosaic"),
                },
            },
            "generation": {
                "multiplier": multiplier,
                "preserve_originals": self.ctrl.get('preserve_originals', ToggleSwitch(None)).get() if 'preserve_originals' in self.ctrl else False,
            },
        }
        return config

    def _on_generate(self):
        config = self._collect_config()
        self.destroy()
        if self.on_generate:
            self.on_generate(config)
