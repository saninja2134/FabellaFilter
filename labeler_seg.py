# Module for labeling images with Segmentation Masks (Polygons).
import cv2
import os
import numpy as np
import shutil

class SegLabeler:
    # An OpenCV-based tool for annotating images with Segmentation Polygons.
    def __init__(self, image_dir="data/sorted/pos", label_dir="data/labels/seg"):
        # Initializes the SegLabeler.
        # Args:
        # image_dir (str): Directory containing images to label.
        # label_dir (str): Directory to save the labels.
        self.image_dir = image_dir
        self.label_dir = label_dir
        image_root = os.path.dirname(os.path.normpath(image_dir))
        image_folder = os.path.basename(os.path.normpath(image_dir))
        self.labeled_dir = os.path.join(image_root, f"{image_folder}_labeled")
        self.unlabeled_dir = os.path.join(image_root, f"{image_folder}_unlabeled")
        
        if os.path.exists(image_dir):
            self.images = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
            self.images.sort()
        else:
            self.images = []
            
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(self.unlabeled_dir, exist_ok=True)
            
        self.index = 0
        self.polygons = [] # List of completed polygons (each is a list of (x,y) tuples)
        self.current_polygon = [] # Points of the polygon currently being drawn
        self.current_image = None
        self.display_src = None # 8-bit BGR source for display
        self.window_name = "YOLO Seg Labeler - [L-Click: Point, R-Click: Close Poly, Space: Save, C: Clear, Z: Undo, Mid-Click: Pan, Scroll: Zoom]"
        
        # Zoom & Pan state
        self.zoom_level = 1.0
        self.offset = [50, 50] # Screen-space offset [x, y]
        self.dragging = False
        self.last_mouse = [0, 0]

        # Display settings (visual only – do not affect saved label files)
        self.PRESET_COLORS = [
            ((0, 255, 0),     "Green"),
            ((0, 255, 255),   "Yellow"),
            ((255, 255, 0),   "Cyan"),
            ((255, 0, 255),   "Magenta"),
            ((0, 0, 255),     "Red"),
            ((0, 128, 255),   "Orange"),
            ((255, 0, 0),     "Blue"),
            ((255, 255, 255), "White"),
        ]
        self.OPACITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.poly_color = (0, 255, 0)   # Active BGR fill colour (visual only)
        self.transparency = 0.3          # Fill alpha
        self.show_direction = False      # Direction marker visibility
        self.show_menu = False           # Menu panel visibility

        self.sync_label_folders()

    def get_image_path(self, image_name):
        return os.path.join(self.image_dir, image_name)

    def get_label_path(self, image_name):
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        return os.path.join(self.label_dir, txt_name)

    def has_saved_label(self, image_name):
        label_path = self.get_label_path(image_name)
        return os.path.exists(label_path) and os.path.getsize(label_path) > 0

    def sync_image_bucket(self, image_name):
        src_path = self.get_image_path(image_name)
        if not os.path.exists(src_path):
            return

        labeled_path = os.path.join(self.labeled_dir, image_name)
        unlabeled_path = os.path.join(self.unlabeled_dir, image_name)
        target_path = labeled_path if self.has_saved_label(image_name) else unlabeled_path
        other_path = unlabeled_path if target_path == labeled_path else labeled_path

        shutil.copy2(src_path, target_path)
        if os.path.exists(other_path):
            os.remove(other_path)

    def sync_label_folders(self):
        for image_name in self.images:
            self.sync_image_bucket(image_name)

    def _outline_color(self, bgr):
        return tuple(int(c * 0.65) for c in bgr)

    def _handle_menu_click(self, x, y):
        # Badge toggle (always-visible, top-right)
        if 1278 <= x <= 1395 and 29 <= y <= 53:
            self.show_menu = not self.show_menu
            return True
        if not self.show_menu:
            return False
        # Panel constants – must mirror the values used in redraw()
        mpx, mpy = 1165, 56
        # Colour swatches
        for i, (color, _) in enumerate(self.PRESET_COLORS):
            cx = mpx + 40 + i * 22
            cy = mpy + 44
            if (x - cx) ** 2 + (y - cy) ** 2 <= 9 ** 2:
                self.poly_color = color
                return True
        # Opacity button
        if mpx <= x <= mpx + 227 and mpy + 62 <= y <= mpy + 90:
            try:
                idx = self.OPACITY_LEVELS.index(self.transparency)
            except ValueError:
                idx = 3
            self.transparency = self.OPACITY_LEVELS[(idx + 1) % len(self.OPACITY_LEVELS)]
            return True
        # Direction button
        if mpx <= x <= mpx + 227 and mpy + 98 <= y <= mpy + 126:
            self.show_direction = not self.show_direction
            return True
        # Absorb any other click inside the panel so it doesn't place a polygon point
        if mpx <= x <= mpx + 232 and mpy <= y <= mpy + 140:
            return True
        return False

    def mouse_callback(self, event, x, y, flags, param):
        # Handles mouse events for drawing, panning, and zooming.
        # Convert screen x,y to image x,y
        img_x = (x - self.offset[0]) / (self.zoom_level if self.zoom_level > 0 else 0.001)
        img_y = (y - self.offset[1]) / (self.zoom_level if self.zoom_level > 0 else 0.001)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._handle_menu_click(x, y):
                self.redraw()
                return
            self.current_polygon.append((img_x, img_y))
            self.redraw()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                self.current_polygon = []
                self.redraw()
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.last_mouse = [x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]
                self.offset[0] += dx
                self.offset[1] += dy
                self.last_mouse = [x, y]
                self.redraw()
                
        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            scale_factor = 1.15 if flags > 0 else 1.0/1.15
            new_zoom = self.zoom_level * scale_factor
            
            # Anchor zoom to mouse pointer
            self.offset[0] = x - (x - self.offset[0]) * scale_factor
            self.offset[1] = y - (y - self.offset[1]) * scale_factor
            self.zoom_level = new_zoom
            self.redraw()

    def redraw(self):
        # Redraws the image and annotations on the canvas.
        if self.display_src is None: return
        
        canvas_h, canvas_w = 950, 1400
        display = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Image dimensions
        h, w = self.display_src.shape[:2]
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        
        if scaled_w > 0 and scaled_h > 0:
            resized = cv2.resize(self.display_src, (scaled_w, scaled_h))
            
            # Intersection calculations
            y1 = max(0, int(self.offset[1]))
            y2 = min(canvas_h, int(self.offset[1]) + scaled_h)
            x1 = max(0, int(self.offset[0]))
            x2 = min(canvas_w, int(self.offset[0]) + scaled_w)
            
            iy1 = max(0, -int(self.offset[1]))
            ix1 = max(0, -int(self.offset[0]))
            iy2 = iy1 + (y2 - y1)
            ix2 = ix1 + (x2 - x1)
            
            if y2 > y1 and x2 > x1 and iy2 > iy1 and ix2 > ix1:
                display[y1:y2, x1:x2] = resized[iy1:iy2, ix1:ix2]

        # Draw Completed Polygons (with alpha blending)
        outline_col = self._outline_color(self.poly_color)
        if self.transparency > 0:
            overlay = display.copy()
            for poly in self.polygons:
                scr_coords = [[int(px * self.zoom_level + self.offset[0]),
                               int(py * self.zoom_level + self.offset[1])] for px, py in poly]
                if scr_coords:
                    cv2.fillPoly(overlay, [np.array(scr_coords, np.int32)], self.poly_color)
            cv2.addWeighted(overlay, self.transparency, display, 1.0 - self.transparency, 0, display)
        for poly in self.polygons:
            scr_coords = [[int(px * self.zoom_level + self.offset[0]),
                           int(py * self.zoom_level + self.offset[1])] for px, py in poly]
            if scr_coords:
                cv2.polylines(display, [np.array(scr_coords, np.int32)], True, outline_col, 2)

        # Direction markers
        if self.show_direction:
            for poly in self.polygons:
                n = len(poly)
                if n == 0:
                    continue
                interval = max(1, n // 5)
                mark_indices = sorted(set(min(k * interval, n - 1) for k in range(5)))
                # Shoelace on image coords to determine winding
                shoelace = sum(
                    (poly[i][0] * poly[(i + 1) % n][1]) - (poly[(i + 1) % n][0] * poly[i][1])
                    for i in range(n)
                )
                winding = "CCW" if shoelace > 0 else "CW"
                # Place winding label above the topmost screen point
                top_sy = min(int(p[1] * self.zoom_level + self.offset[1]) for p in poly)
                top_sx = int(sum(p[0] for p in poly) / n * self.zoom_level + self.offset[0])
                cv2.putText(display, winding, (top_sx - 15, top_sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 2)
                for pt_idx in mark_indices:
                    sx = int(poly[pt_idx][0] * self.zoom_level + self.offset[0])
                    sy = int(poly[pt_idx][1] * self.zoom_level + self.offset[1])
                    vertex_label = str(pt_idx + 1)
                    text_x = sx - 4 * len(vertex_label)
                    cv2.circle(display, (sx, sy), 8, (255, 255, 255), -1)
                    cv2.circle(display, (sx, sy), 8, (0, 0, 0), 1)
                    cv2.putText(display, vertex_label, (text_x, sy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw Current Polygon
        if self.current_polygon:
            scr_coords = []
            for px, py in self.current_polygon:
                sx = int(px * self.zoom_level + self.offset[0])
                sy = int(py * self.zoom_level + self.offset[1])
                scr_coords.append([sx, sy])
                if 0 <= sx < canvas_w and 0 <= sy < canvas_h:
                    cv2.circle(display, (sx, sy), 4, (0, 0, 255), -1)
            
            if len(scr_coords) > 1:
                cv2.polylines(display, [np.array(scr_coords, np.int32)], False, (255, 0, 0), 2)

        # UI Overlays
        info = f"Image: {self.index + 1}/{len(self.images)} | Zoom: {self.zoom_level:.2f}x | {self.images[self.index]}"
        cv2.putText(display, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "L-Click: Point | R-Click: Close Poly | Space: Save | C: Clear | Z: Undo | A/D: Nav | M: Menu | T: Opacity | V: Direction",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Menu badge (always visible, top-right)
        bx1, by1, bx2, by2 = 1278, 29, 1395, 53
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (150, 150, 150), 1)
        badge_label = "MENU [open]" if self.show_menu else "MENU"
        cv2.putText(display, badge_label, (bx1 + 6, by2 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Menu panel
        if self.show_menu:
            mpx, mpy, mpw, mph = 1165, 56, 232, 140
            panel_overlay = display.copy()
            cv2.rectangle(panel_overlay, (mpx, mpy), (mpx + mpw, mpy + mph), (35, 35, 35), -1)
            cv2.addWeighted(panel_overlay, 0.90, display, 0.10, 0, display)
            cv2.rectangle(display, (mpx, mpy), (mpx + mpw, mpy + mph), (130, 130, 130), 1)
            cv2.putText(display, "DISPLAY SETTINGS", (mpx + 8, mpy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            # Colour swatches
            cv2.putText(display, "Col:", (mpx + 5, mpy + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
            for i, (color, _) in enumerate(self.PRESET_COLORS):
                cx = mpx + 40 + i * 22
                cy = mpy + 44
                cv2.circle(display, (cx, cy), 9, color, -1)
                cv2.circle(display, (cx, cy), 9, (80, 80, 80), 1)
                if color == self.poly_color:
                    cv2.circle(display, (cx, cy), 11, (255, 255, 255), 2)
            # Opacity button
            op_y1, op_y2 = mpy + 62, mpy + 88
            cv2.rectangle(display, (mpx + 5, op_y1), (mpx + mpw - 5, op_y2), (60, 60, 60), -1)
            cv2.rectangle(display, (mpx + 5, op_y1), (mpx + mpw - 5, op_y2), (110, 110, 110), 1)
            cv2.putText(display, f"Opacity: {int(self.transparency * 100)}%",
                        (mpx + 55, op_y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
            # Direction button
            dir_y1, dir_y2 = mpy + 98, mpy + 124
            dir_col = (45, 90, 45) if self.show_direction else (60, 60, 60)
            cv2.rectangle(display, (mpx + 5, dir_y1), (mpx + mpw - 5, dir_y2), dir_col, -1)
            cv2.rectangle(display, (mpx + 5, dir_y1), (mpx + mpw - 5, dir_y2), (110, 110, 110), 1)
            dir_text = "Direction: ON" if self.show_direction else "Direction: OFF"
            cv2.putText(display, dir_text, (mpx + 45, dir_y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)

        cv2.imshow(self.window_name, display)

    def save_label(self):
        # Saves the current segmentation labels to a text file.
        if not self.polygons or self.current_image is None: 
            print("No completed polygons to save.")
            return
            
        im_h, im_w = self.current_image.shape[:2]
        txt_name = os.path.splitext(self.images[self.index])[0] + ".txt"

        with open(self.get_label_path(self.images[self.index]), 'w') as f:
            for poly in self.polygons:
                flat = []
                for x, y in poly:
                    # Clamp coordinates to 0-1 range
                    nx = max(0.0, min(1.0, x / im_w))
                    ny = max(0.0, min(1.0, y / im_h))
                    flat.append(nx)
                    flat.append(ny)
                f.write(f"0 {' '.join([f'{v:.6f}' for v in flat])}\n")

            self.sync_image_bucket(self.images[self.index])
                
        print(f"Saved: {txt_name}")

    def load_existing_labels(self):
        # Loads existing labels if they exist for the current image.
        self.polygons = []
        self.current_polygon = []
        
        image_name = self.images[self.index]
        txt_path = self.get_label_path(image_name)
        self.sync_image_bucket(image_name)
        
        if not os.path.exists(txt_path) or self.current_image is None:
            return
            
        im_h, im_w = self.current_image.shape[:2]
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        # parts[0] is class id
                        coords = [float(p) for p in parts[1:]]
                        poly = []
                        for i in range(0, len(coords), 2):
                            x = coords[i] * im_w
                            y = coords[i+1] * im_h
                            poly.append((x, y))
                        if poly:
                            self.polygons.append(poly)
        except Exception as e:
            print(f"Error loading existing label: {e}")

    def run(self):
        # Starts the labeling session.
        if not self.images:
            print(f"No images found in {self.image_dir}")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while 0 <= self.index < len(self.images):
            img_path = os.path.join(self.image_dir, self.images[self.index])
            self.current_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                self.index += 1
                continue
            
            # 8-bit conversion for display
            if self.current_image.dtype == np.uint16:
                disp = (self.current_image / 256).astype(np.uint8)
            else:
                disp = self.current_image
            
            # Auto-contrast for bone visibility
            if disp is not None:
                dmin, dmax = disp.min(), disp.max()
                if dmax > dmin:
                    disp = ((disp - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                
                if len(disp.shape) == 2:
                    self.display_src = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                elif len(disp.shape) == 3 and disp.shape[2] == 1:
                    self.display_src = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                elif len(disp.shape) == 3 and disp.shape[2] == 4:
                    self.display_src = cv2.cvtColor(disp, cv2.COLOR_BGRA2BGR)
                else:
                    self.display_src = disp
            
            self.load_existing_labels()
            
            # Initial framing
            if self.index >= 0 and self.zoom_level == 1.0:
                 self.zoom_level, self.offset = 0.6, [200, 50]

            self.redraw()
            
            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == 27 or key == ord('q'): # ESC or Q
                    cv2.destroyAllWindows()
                    return
                elif key == ord('d') or key == 83: # Right arrow or D
                    self.index += 1
                    break
                elif key == ord('a') or key == 81: # Left arrow or A
                    self.index = max(0, self.index - 1)
                    break
                elif key == 32: # Space
                    self.save_label()
                    self.index += 1
                    break
                elif key == ord('c'): # Clear current
                    self.current_polygon = []
                    self.redraw()
                elif key == ord('z'): # Undo
                    if self.current_polygon:
                        self.current_polygon.pop()
                    elif self.polygons:
                        self.polygons.pop()
                    self.redraw()
                elif key == ord('m'): # Toggle menu panel
                    self.show_menu = not self.show_menu
                    self.redraw()
                elif key == ord('t'): # Cycle opacity
                    try:
                        idx = self.OPACITY_LEVELS.index(self.transparency)
                    except ValueError:
                        idx = 3
                    self.transparency = self.OPACITY_LEVELS[(idx + 1) % len(self.OPACITY_LEVELS)]
                    self.redraw()
                elif key == ord('v'): # Toggle direction markers
                    self.show_direction = not self.show_direction
                    self.redraw()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = SegLabeler()
    labeler.run()
