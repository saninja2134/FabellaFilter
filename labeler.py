import cv2
import os
import numpy as np

class OBBLabeler:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        self.images.sort()
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            
        self.index = 0
        self.points = [] # Image coordinates
        self.current_image = None
        self.display_src = None # 8-bit BGR source for display
        self.window_name = "YOLO OBB Labeler - [Arrows/WASD: Nav, Space: Save, C: Clear, Middle Click: Pan, Scroll: Zoom, Q: Quit]"
        
        # Zoom & Pan state
        self.zoom_level = 1.0
        self.offset = [50, 50] # Screen-space offset [x, y]
        self.dragging = False
        self.last_mouse = [0, 0]
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # Convert screen x,y to image x,y
        img_x = (x - self.offset[0]) / (self.zoom_level if self.zoom_level > 0 else 0.001)
        img_y = (y - self.offset[1]) / (self.zoom_level if self.zoom_level > 0 else 0.001)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 3:
                self.points.append((img_x, img_y))
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

    def get_obb_coords(self, p1, p2, p3):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0: return None
        
        ux, uy = dx/length, dy/length
        vx, vy = -uy, ux
        dist = (p3[0] - p1[0]) * vx + (p3[1] - p1[1]) * vy
        
        return [(p1[0], p1[1]), (p2[0], p2[1]), 
                (p2[0] + dist * vx, p2[1] + dist * vy), 
                (p1[0] + dist * vx, p1[1] + dist * vy)]

    def redraw(self):
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

        # Draw Points/Shapes
        for p in self.points:
            sx = int(p[0] * self.zoom_level + self.offset[0])
            sy = int(p[1] * self.zoom_level + self.offset[1])
            if 0 <= sx < canvas_w and 0 <= sy < canvas_h:
                cv2.circle(display, (sx, sy), 5, (0, 0, 255), -1)

        if len(self.points) == 2:
            p1s = (int(self.points[0][0] * self.zoom_level + self.offset[0]),
                   int(self.points[0][1] * self.zoom_level + self.offset[1]))
            p2s = (int(self.points[1][0] * self.zoom_level + self.offset[0]),
                   int(self.points[1][1] * self.zoom_level + self.offset[1]))
            cv2.line(display, p1s, p2s, (255, 0, 0), 2)
            
        elif len(self.points) == 3:
            coords = self.get_obb_coords(*self.points)
            if coords:
                scr_coords = []
                for cx, cy in coords:
                    scr_coords.append([int(cx * self.zoom_level + self.offset[0]),
                                     int(cy * self.zoom_level + self.offset[1])])
                cv2.polylines(display, [np.array(scr_coords, np.int32)], True, (0, 255, 0), 2)
        
        # UI Overlays
        info = f"Image: {self.index + 1}/{len(self.images)} | Zoom: {self.zoom_level:.2f}x | {self.images[self.index]}"
        cv2.putText(display, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "L-Click: Point | Scroll: Zoom | Mid-Click: Pan | A/D: Nav | Space: Save", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(self.window_name, display)

    def save_label(self):
        if len(self.points) != 3 or self.current_image is None: return
        coords = self.get_obb_coords(*self.points)
        if coords is None: return
        
        im_h, im_w = self.current_image.shape[:2]
        flat = []
        for x, y in coords:
            flat.append(x / im_w)
            flat.append(y / im_h)
        txt_name = os.path.splitext(self.images[self.index])[0] + ".txt"
        with open(os.path.join(self.label_dir, txt_name), 'w') as f:
            f.write(f"0 {' '.join([f'{v:.6f}' for v in flat])}\n")
        print(f"Saved: {txt_name}")

    def run(self):
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
                self.display_src = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR) if len(disp.shape) == 2 else disp
            
            self.points = []
            
            # Initial framing
            if self.index >= 0 and self.zoom_level == 1.0:
                 self.zoom_level, self.offset = 0.6, [200, 50]

            self.redraw()
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'): 
                    cv2.destroyAllWindows()
                    return
                elif k == ord(' '): # Space: Save and Next
                    if len(self.points) == 3: self.save_label()
                    self.index += 1
                    break
                elif k == ord('d'): # D: Next
                    self.index += 1
                    break
                elif k == ord('a'): # A: Previous
                    self.index = max(0, self.index - 1)
                    break
                elif k == ord('c'): # Clear points
                    self.points = []
                    self.redraw()

        print("Labelling session finished.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder = "dataset_sorted/pos"
    label_folder = "labels/pos"
    labeler = OBBLabeler(image_folder, label_folder)
    labeler.run()
