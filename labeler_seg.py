# Module for labeling images with Segmentation Masks (Polygons).
import cv2
import os
import numpy as np

class SegLabeler:
    # An OpenCV-based tool for annotating images with Segmentation Polygons.
    def __init__(self, image_dir="data/sorted/pos", label_dir="data/labels/seg"):
        # Initializes the SegLabeler.
        # 
        # Args:
        # image_dir (str): Directory containing images to label.
        # label_dir (str): Directory to save the labels.
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        if os.path.exists(image_dir):
            self.images = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
            self.images.sort()
        else:
            self.images = []
            
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            
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

    def mouse_callback(self, event, x, y, flags, param):
        # Handles mouse events for drawing, panning, and zooming.
        # Convert screen x,y to image x,y
        img_x = (x - self.offset[0]) / (self.zoom_level if self.zoom_level > 0 else 0.001)
        img_y = (y - self.offset[1]) / (self.zoom_level if self.zoom_level > 0 else 0.001)

        if event == cv2.EVENT_LBUTTONDOWN:
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
        overlay = display.copy()
        for poly in self.polygons:
            scr_coords = []
            for px, py in poly:
                scr_coords.append([int(px * self.zoom_level + self.offset[0]),
                                   int(py * self.zoom_level + self.offset[1])])
            if scr_coords:
                pts = np.array(scr_coords, np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.polylines(display, [pts], True, (0, 200, 0), 2)
                
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

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
        cv2.putText(display, "L-Click: Point | R-Click: Close Poly | Space: Save | C: Clear | Z: Undo | A/D: Nav", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(self.window_name, display)

    def save_label(self):
        # Saves the current segmentation labels to a text file.
        if not self.polygons or self.current_image is None: 
            print("No completed polygons to save.")
            return
            
        im_h, im_w = self.current_image.shape[:2]
        txt_name = os.path.splitext(self.images[self.index])[0] + ".txt"
        
        with open(os.path.join(self.label_dir, txt_name), 'w') as f:
            for poly in self.polygons:
                flat = []
                for x, y in poly:
                    # Clamp coordinates to 0-1 range
                    nx = max(0.0, min(1.0, x / im_w))
                    ny = max(0.0, min(1.0, y / im_h))
                    flat.append(nx)
                    flat.append(ny)
                f.write(f"0 {' '.join([f'{v:.6f}' for v in flat])}\n")
                
        print(f"Saved: {txt_name}")

    def load_existing_labels(self):
        # Loads existing labels if they exist for the current image.
        self.polygons = []
        self.current_polygon = []
        
        txt_name = os.path.splitext(self.images[self.index])[0] + ".txt"
        txt_path = os.path.join(self.label_dir, txt_name)
        
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

        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = SegLabeler()
    labeler.run()
