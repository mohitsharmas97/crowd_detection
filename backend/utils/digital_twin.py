import numpy as np
import cv2

class DigitalTwin:
    """Create a basic digital twin visualization of the crowd"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.width = frame_width
        self.height = frame_height
        self.person_positions = []
        self.person_velocities = []
        self.trail_history = []  # Movement trails
        self.max_trail_length = 30
        
    def update(self, person_centers, movement_vectors=None):
        """
        Update digital twin state
        
        Args:
            person_centers: List of (x, y) positions
            movement_vectors: Optional movement vectors for each person
        """
        self.person_positions = person_centers
        
        if movement_vectors is not None and len(movement_vectors) > 0:
            self.person_velocities = movement_vectors
        else:
            self.person_velocities = [(0, 0)] * len(person_centers)
        
        # Update movement trails
        self.trail_history.append(person_centers.copy() if person_centers else [])
        if len(self.trail_history) > self.max_trail_length:
            self.trail_history.pop(0)
    
    def render(self, show_trails=True, show_velocities=True):
        """
        Render the digital twin visualization
        
        Returns:
            numpy array: Rendered digital twin image
        """
        # Create blank canvas with dark background
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas.fill(20)  # Dark gray background
        
        # Draw grid
        self._draw_grid(canvas)
        
        # Draw movement trails
        if show_trails and len(self.trail_history) > 1:
            self._draw_trails(canvas)
        
        # Draw people
        for i, (x, y) in enumerate(self.person_positions):
            # Person dot
            cv2.circle(canvas, (int(x), int(y)), 8, (0, 255, 255), -1)
            cv2.circle(canvas, (int(x), int(y)), 10, (255, 255, 255), 1)
            
            # Velocity vector
            if show_velocities and i < len(self.person_velocities):
                vx, vy = self.person_velocities[i]
                end_x = int(x + vx * 20)
                end_y = int(y + vy * 20)
                cv2.arrowedLine(canvas, (int(x), int(y)), (end_x, end_y),
                              (0, 255, 0), 2, tipLength=0.3)
        
        # Draw info overlay
        self._draw_info_overlay(canvas)
        
        return canvas
    
    def _draw_grid(self, canvas):
        """Draw a subtle grid on the canvas"""
        grid_spacing = 50
        grid_color = (40, 40, 40)
        
        # Vertical lines
        for x in range(0, self.width, grid_spacing):
            cv2.line(canvas, (x, 0), (x, self.height), grid_color, 1)
        
        # Horizontal lines
        for y in range(0, self.height, grid_spacing):
            cv2.line(canvas, (0, y), (self.width, y), grid_color, 1)
    
    def _draw_trails(self, canvas):
        """Draw movement trails for each person"""
        num_people = len(self.person_positions)
        
        # Draw trails with fading effect
        for trail_idx, positions in enumerate(self.trail_history):
            alpha = trail_idx / len(self.trail_history)  # Fade based on age
            color_intensity = int(150 * alpha)
            color = (color_intensity, color_intensity, color_intensity)
            
            for x, y in positions:
                cv2.circle(canvas, (int(x), int(y)), 2, color, -1)
    
    def _draw_info_overlay(self, canvas):
        """Draw information overlay"""
        info_text = f'People: {len(self.person_positions)}'
        
        # Semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (5, 5), (200, 35), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Text
        cv2.putText(canvas, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def get_crowd_distribution_map(self):
        """Get 2D distribution map of crowd"""
        distribution = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Create gaussian blobs at each person position
        for x, y in self.person_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create a gaussian kernel
                size = 30
                sigma = 10
                kernel = self._gaussian_kernel(size, sigma)
                
                # Add to distribution map
                x_start = max(0, int(x) - size // 2)
                x_end = min(self.width, int(x) + size // 2)
                y_start = max(0, int(y) - size // 2)
                y_end = min(self.height, int(y) + size // 2)
                
                k_x_start = size // 2 - (int(x) - x_start)
                k_x_end = k_x_start + (x_end - x_start)
                k_y_start = size // 2 - (int(y) - y_start)
                k_y_end = k_y_start + (y_end - y_start)
                
                distribution[y_start:y_end, x_start:x_end] += kernel[k_y_start:k_y_end, k_x_start:k_x_end]
        
        return distribution
    
    def _gaussian_kernel(self, size, sigma):
        """Create a 2D gaussian kernel"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.sum()
    
    def export_state(self):
        """Export current state as JSON-serializable dict"""
        return {
            'positions': [[int(x), int(y)] for x, y in self.person_positions],
            'velocities': [[float(vx), float(vy)] for vx, vy in self.person_velocities],
            'count': len(self.person_positions)
        }
