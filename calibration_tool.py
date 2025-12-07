#!/usr/bin/env python3
"""
Calibration Tool for Parking Space Detection
This tool allows users to:
1. Capture frames from the live camera
2. Draw polygonal regions for different parking areas
3. Define ignore regions (buildings, roads, etc.)
4. Generate a probability map
5. Save regions to regions.json
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
import requests
from PIL import Image, ImageTk, ImageDraw
import io
import os

class CalibrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Space Calibration Tool")
        self.root.geometry("1400x900")
        
        # Camera settings
        self.camera_url = "http://170.249.152.2:8080/video.mjpg"
        self.current_frame = None
        self.current_image = None
        self._cap = None  # persistent VideoCapture for MJPG/RTSP streams
        
        # Drawing state
        self.current_region = []
        self.current_region_name = ""
        self.regions = {}
        self.drawing = False
        
        # Scaling
        self.scale_factor = 1.0
        self.display_width = 800
        self.display_height = 600
        # Displayed image placement (for centered image inside label)
        self.offset_x = 0
        self.offset_y = 0
        self.last_display_size = (0, 0)
        
        # Region names and descriptions
        self.region_info = {
            'upper_level_l': 'Upper Level Left',
            'upper_level_m': 'Upper Level Middle', 
            'upper_level_r': 'Upper Level Right',
            'close_perp': 'Close Perpendicular',
            'far_side': 'Far Side',
            'close_side': 'Close Side',
            'far_perp': 'Far Perpendicular',
            'small_park': 'Small Parking Area',
            'ignore_regions': 'Areas to Ignore (buildings, roads)'
        }
        
        self.setup_gui()
        self.load_existing_regions()
        
    def _iter_polygons(self, region_points):
        """Yield one or more polygons from a region entry.
        Accepts either a single polygon: [(x,y), ...]
        or a list of polygons: [[(x,y),...], [(x,y),...], ...]
        """
        if not region_points:
            return []
        # A point is a tuple/list of 2 numbers
        def _is_point(p):
            return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)
        # Single polygon
        if isinstance(region_points, (list, tuple)) and region_points and _is_point(region_points[0]):
            return [region_points]
        # List of polygons
        if isinstance(region_points, (list, tuple)) and region_points and isinstance(region_points[0], (list, tuple)):
            # Filter only valid polygons
            polys = []
            for poly in region_points:
                if poly and isinstance(poly, (list, tuple)) and _is_point(poly[0]):
                    polys.append(poly)
            return polys
        return []
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left control panel with scrollbar
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # Create canvas for scrollable controls
        canvas = tk.Canvas(left_frame, width=320)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Calibration Tool", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(scrollable_frame, text="Camera Controls", padding=10)
        camera_frame.pack(fill='x', pady=(0, 10))
        
        self.capture_btn = ttk.Button(camera_frame, text="Capture Frame", command=self.capture_frame)
        self.capture_btn.pack(fill='x', pady=2)
        
        self.refresh_btn = ttk.Button(camera_frame, text="Refresh Frame", command=self.refresh_frame)
        self.refresh_btn.pack(fill='x', pady=2)
        
        # Region drawing controls
        drawing_frame = ttk.LabelFrame(scrollable_frame, text="Region Drawing", padding=10)
        drawing_frame.pack(fill='x', pady=(0, 10))
        
        # Region name input
        ttk.Label(drawing_frame, text="Region Name:").pack(anchor='w')
        self.region_name_var = tk.StringVar()
        self.region_name_entry = ttk.Entry(drawing_frame, textvariable=self.region_name_var)
        self.region_name_entry.pack(fill='x', pady=(0, 10))
        
        # Quick region selection buttons
        quick_frame = ttk.Frame(drawing_frame)
        quick_frame.pack(fill='x', pady=(0, 10))
        
        for region_name, description in self.region_info.items():
            btn = ttk.Button(quick_frame, text=description, 
                           command=lambda r=region_name: self.select_region(r))
            btn.pack(fill='x', pady=1)
        
        # Drawing controls
        control_frame = ttk.Frame(drawing_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        self.start_draw_btn = ttk.Button(control_frame, text="Start Drawing", command=self.start_drawing)
        self.start_draw_btn.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        self.stop_draw_btn = ttk.Button(control_frame, text="Stop Drawing", command=self.stop_drawing, state='disabled')
        self.stop_draw_btn.pack(side='right', fill='x', expand=True, padx=(2, 0))
        
        # Region management
        region_mgmt_frame = ttk.LabelFrame(scrollable_frame, text="Region Management", padding=10)
        region_mgmt_frame.pack(fill='x', pady=(0, 10))
        
        self.clear_current_btn = ttk.Button(region_mgmt_frame, text="Clear Current Region", command=self.clear_current_region)
        self.clear_current_btn.pack(fill='x', pady=2)
        
        self.undo_btn = ttk.Button(region_mgmt_frame, text="Undo Last Point", command=self.undo_last_point)
        self.undo_btn.pack(fill='x', pady=2)
        
        self.delete_btn = ttk.Button(region_mgmt_frame, text="Delete Selected Region", command=self.delete_selected_region)
        self.delete_btn.pack(fill='x', pady=2)
        
        self.clear_all_btn = ttk.Button(region_mgmt_frame, text="Clear All Regions", command=self.clear_all_regions)
        self.clear_all_btn.pack(fill='x', pady=2)
        
        # File operations
        file_frame = ttk.LabelFrame(scrollable_frame, text="File Operations", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        self.save_btn = ttk.Button(file_frame, text="Save Regions", command=self.save_regions)
        self.save_btn.pack(fill='x', pady=2)
        
        self.load_btn = ttk.Button(file_frame, text="Load Regions", command=self.load_regions)
        self.load_btn.pack(fill='x', pady=2)
        
        # Probability map generation
        prob_frame = ttk.LabelFrame(scrollable_frame, text="Probability Map", padding=10)
        prob_frame.pack(fill='x', pady=(0, 10))
        
        self.generate_prob_btn = ttk.Button(prob_frame, text="Generate Probability Map", command=self.generate_probability_map)
        self.generate_prob_btn.pack(fill='x', pady=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var, font=('Arial', 10))
        status_label.pack(pady=(10, 0))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Right side - image display
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Image display on a Canvas (top-left anchored for precise clicks)
        self.canvas = tk.Canvas(right_frame, width=self.display_width, height=self.display_height, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        # Recompute scale when the canvas is resized
        self.canvas.bind("<Configure>", self.handle_canvas_resize)

    def handle_canvas_resize(self, event):
        """Update canvas target size and redraw image to keep clicks aligned."""
        self.display_width = max(1, event.width)
        self.display_height = max(1, event.height)
        if self.current_image is not None:
            self.display_image()
        
    def capture_frame(self):
        """Capture a frame from the camera"""
        try:
            self.status_var.set("Capturing frame...")
            url = self.camera_url
            # Use VideoCapture for continuous streams like MJPG/RTSP
            if any(x in url.lower() for x in ['.mjpg', 'rtsp://']) and 'webcapture.jpg' not in url:
                if self._cap is None or not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(url)
                if self._cap is not None and self._cap.isOpened():
                    ok, frame = self._cap.read()
                    if ok and frame is not None:
                        self.current_frame = frame
                        # Convert BGR to RGB for PIL
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.current_image = Image.fromarray(rgb)
                        self.display_image()
                        self.status_var.set("Frame captured successfully")
                    else:
                        messagebox.showerror("Error", "Failed to read frame from stream")
                        self.status_var.set("Failed to capture frame")
                else:
                    messagebox.showerror("Error", "Unable to open video stream")
                    self.status_var.set("Failed to capture frame")
            else:
                # Snapshot URL
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    self.current_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    self.current_image = image
                    self.display_image()
                    self.status_var.set("Frame captured successfully")
                else:
                    messagebox.showerror("Error", f"Failed to capture frame: HTTP {response.status_code}")
                    self.status_var.set("Failed to capture frame")
        except Exception as e:
            messagebox.showerror("Error", f"Error capturing frame: {e}")
            self.status_var.set("Error capturing frame")
    
    def refresh_frame(self):
        """Refresh the current frame"""
        if self.current_frame is not None:
            self.display_image()
            self.status_var.set("Frame refreshed")
        else:
            messagebox.showwarning("Warning", "No frame to refresh. Capture a frame first.")
    
    def display_image(self):
        """Display the current image with regions drawn"""
        if self.current_image is None:
            return
            
        # First resize the image to fit display area
        display_image = self.resize_image_to_fit(self.current_image.copy())
        
        # Create a drawing context on the scaled image
        draw = ImageDraw.Draw(display_image)
        
        # Draw all regions (scaled coordinates)
        for region_name, points in self.regions.items():
            polygons = self._iter_polygons(points)
            for poly in polygons:
                if len(poly) < 2:
                    continue
                # Choose color based on region type
                if region_name == 'ignore_regions':
                    color = (255, 0, 0)  # Red for ignore regions
                else:
                    color = (0, 255, 0)  # Green for parking regions
                
                # Scale the points for display
                scaled_points = [(int(p[0] * self.scale_factor), int(p[1] * self.scale_factor)) for p in poly]
                
                # Draw lines between points
                for i in range(len(scaled_points) - 1):
                    draw.line([scaled_points[i], scaled_points[i + 1]], fill=color, width=3)
                
                # Close the polygon
                if len(scaled_points) >= 3:
                    draw.line([scaled_points[-1], scaled_points[0]], fill=color, width=3)
                
                # Draw points
                for point in scaled_points:
                    draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill=color)
                
                # Add region name label
                if scaled_points:
                    center_x = sum(p[0] for p in scaled_points) // len(scaled_points)
                    center_y = sum(p[1] for p in scaled_points) // len(scaled_points)
                    draw.text((center_x, center_y), region_name, fill=color)
        
        # Draw current region being drawn (scaled coordinates)
        if self.current_region:
            color = (0, 0, 255)  # Blue for current region
            scaled_current_points = [(int(p[0] * self.scale_factor), int(p[1] * self.scale_factor)) for p in self.current_region]
            # Draw segments and points
            if len(scaled_current_points) >= 2:
                for i in range(len(scaled_current_points) - 1):
                    draw.line([scaled_current_points[i], scaled_current_points[i + 1]], fill=color, width=3)
            for point in scaled_current_points:
                draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill=color)
        
        # Convert to PhotoImage and display on canvas (top-left anchored)
        self.canvas_image_photo = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_image_photo, anchor='nw')
        # Store displayed image metrics (no offsets on canvas)
        disp_w, disp_h = display_image.size
        self.offset_x = 0
        self.offset_y = 0
        self.last_display_size = (disp_w, disp_h)
    
    def resize_image_to_fit(self, image):
        """Resize image to fit the display area while maintaining aspect ratio"""
        # Get original image dimensions
        img_width, img_height = image.size
        
        # Determine current canvas size (fallback to defaults before layout)
        canvas_w = self.canvas.winfo_width() if hasattr(self, 'canvas') else self.display_width
        canvas_h = self.canvas.winfo_height() if hasattr(self, 'canvas') else self.display_height
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = self.display_width
            canvas_h = self.display_height
        # Calculate scaling factor to fit the image
        scale_x = canvas_w / img_width
        scale_y = canvas_h / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
        
        # Calculate new dimensions
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        # Resize the image
        if self.scale_factor < 1.0:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # If not scaling, ensure canvas knows the image size for accurate clicks
            self.last_display_size = (img_width, img_height)
        
        return image
    
    def convert_display_to_image_coords(self, display_x, display_y):
        """Convert display coordinates back to original image coordinates"""
        if self.current_image is None:
            return display_x, display_y
        
        # Adjust for centered image offsets inside the label
        adj_x = display_x - self.offset_x
        adj_y = display_y - self.offset_y
        # If click is outside the displayed image, clamp to edge
        disp_w, disp_h = self.last_display_size
        if disp_w and disp_h:
            adj_x = min(max(adj_x, 0), disp_w - 1)
            adj_y = min(max(adj_y, 0), disp_h - 1)
        
        # Convert back to original coordinates
        if self.scale_factor < 1.0:
            orig_x = int(adj_x / self.scale_factor)
            orig_y = int(adj_y / self.scale_factor)
            return orig_x, orig_y
        else:
            return adj_x, adj_y
    
    def select_region(self, region_name):
        """Select a region for drawing"""
        self.current_region_name = region_name
        self.region_name_var.set(region_name)
        self.status_var.set(f"Selected region: {self.region_info.get(region_name, region_name)}")
    
    def start_drawing(self):
        """Start drawing mode"""
        if not self.current_region_name:
            messagebox.showwarning("Warning", "Please select a region name first")
            return
        
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please capture a frame first")
            return
        
        self.drawing = True
        self.current_region = []
        self.start_draw_btn.config(state='disabled')
        self.stop_draw_btn.config(state='normal')
        self.status_var.set("Drawing mode active - click on image to add points")
    
    def stop_drawing(self):
        """Stop drawing mode and save region"""
        if not self.drawing:
            return
        
        self.drawing = False
        self.start_draw_btn.config(state='normal')
        self.stop_draw_btn.config(state='disabled')
        
        if len(self.current_region) >= 3:
            # Save the region as a separate polygon (supports multiple polys per region)
            if self.current_region_name not in self.regions or not self.regions[self.current_region_name]:
                self.regions[self.current_region_name] = [list(self.current_region)]
            else:
                existing = self.regions[self.current_region_name]
                # If existing is a single polygon (flat), upgrade to list of polygons
                if isinstance(existing, list) and existing and isinstance(existing[0], (list, tuple)) and len(existing[0]) == 2:
                    self.regions[self.current_region_name] = [existing, list(self.current_region)]
                else:
                    # Already a list of polygons
                    self.regions[self.current_region_name].append(list(self.current_region))
            self.current_region = []
            self.status_var.set(f"Region {self.current_region_name} saved with {len(self.regions[self.current_region_name])} points")
            self.display_image()
        else:
            messagebox.showwarning("Warning", "Region must have at least 3 points")
            self.current_region = []
    
    def on_mouse_click(self, event):
        """Handle mouse clicks for drawing"""
        if not self.drawing:
            return
        
        # Convert display coordinates to original image coordinates
        x, y = self.convert_display_to_image_coords(event.x, event.y)
        
        # Add point to current region
        self.current_region.append((x, y))
        self.status_var.set(f"Added point ({x}, {y}) - Total points: {len(self.current_region)}")
        self.display_image()
    
    def clear_current_region(self):
        """Clear the current region being drawn"""
        self.current_region = []
        self.display_image()
        self.status_var.set("Current region cleared")
    
    def undo_last_point(self):
        """Remove the last point from current region"""
        if self.current_region:
            removed = self.current_region.pop()
            self.status_var.set(f"Removed point {removed}")
            self.display_image()
        else:
            self.status_var.set("No points to remove")
    
    def delete_selected_region(self):
        """Delete the selected region"""
        if self.current_region_name and self.current_region_name in self.regions:
            del self.regions[self.current_region_name]
            self.status_var.set(f"Deleted region: {self.current_region_name}")
            self.display_image()
        else:
            messagebox.showwarning("Warning", "Please select a region to delete")
    
    def clear_all_regions(self):
        """Clear all regions"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all regions?"):
            self.regions = {}
            self.current_region = []
            self.display_image()
            self.status_var.set("All regions cleared")
    
    def save_regions(self):
        """Save regions to regions.json"""
        try:
            # Convert regions to the format expected by the system
            regions_data = {}
            
            for region_name, points in self.regions.items():
                if region_name == 'ignore_regions':
                    # Ensure ignore_regions is a list of polygons
                    polys = self._iter_polygons(points)
                    regions_data['ignore_regions'] = [list(map(list, poly)) for poly in polys]
                else:
                    # Save first polygon if multiple, or the single polygon
                    polys = self._iter_polygons(points)
                    if polys:
                        regions_data[region_name] = list(map(list, polys[0]))
            
            # Save to file
            with open('regions.json', 'w') as f:
                json.dump(regions_data, f, indent=2)
            
            self.status_var.set("Regions saved to regions.json")
            messagebox.showinfo("Success", "Regions saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving regions: {e}")
            self.status_var.set("Error saving regions")
    
    def load_regions(self):
        """Load regions from regions.json"""
        try:
            if os.path.exists('regions.json'):
                with open('regions.json', 'r') as f:
                    data = json.load(f)
                
                self.regions = {}
                
                for region_name, points in data.items():
                    if region_name == 'ignore_regions':
                        # Store as list of polygons
                        if isinstance(points, list) and points and isinstance(points[0], list) and len(points[0]) >= 3:
                            self.regions['ignore_regions'] = [p for p in points]
                        else:
                            self.regions['ignore_regions'] = []
                    else:
                        # Store as single polygon (list of points)
                        if isinstance(points, list) and points and isinstance(points[0], list) and len(points) >= 3:
                            self.regions[region_name] = [points]
                
                self.display_image()
                self.status_var.set("Regions loaded from regions.json")
                messagebox.showinfo("Success", "Regions loaded successfully!")
            else:
                messagebox.showwarning("Warning", "regions.json not found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading regions: {e}")
            self.status_var.set("Error loading regions")
    
    def load_existing_regions(self):
        """Load existing regions if they exist"""
        if os.path.exists('regions.json'):
            self.load_regions()
    
    def generate_probability_map(self):
        """Generate a probability map based on the defined regions"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please capture a frame first")
            return
        
        if not self.regions:
            messagebox.showwarning("Warning", "Please define some regions first")
            return
        
        try:
            self.status_var.set("Generating probability map...")
            
            # Create probability map
            height, width = self.current_frame.shape[:2]
            prob_map = np.zeros((height, width), dtype=np.uint8)
            
            # For each parking region, add probability
            for region_name, points in self.regions.items():
                if region_name == 'ignore_regions':
                    continue
                polygons = self._iter_polygons(points)
                for poly in polygons:
                    if len(poly) < 3:
                        continue
                    # Clip coordinates into image bounds
                    poly_np = np.array([[min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)] for x, y in poly], dtype=np.int32)
                    if poly_np.shape[0] >= 3:
                        # Create mask for this polygon
                        mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_np], 255)
                        prob_map = cv2.add(prob_map, mask)
            
            # Normalize to 0-255 range
            if prob_map.max() > 0:
                prob_map = (prob_map * 255 / prob_map.max()).astype(np.uint8)
            
            # Save probability map
            cv2.imwrite('probability_map_new.png', prob_map)
            
            self.status_var.set("Probability map generated and saved")
            messagebox.showinfo("Success", "Probability map generated and saved to probability_map_new.png!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating probability map: {e}")
            self.status_var.set("Error generating probability map")

def main():
    # Create Demo directory if it doesn't exist
    if not os.path.exists('Demo'):
        os.makedirs('Demo')
    
    root = tk.Tk()
    app = CalibrationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()