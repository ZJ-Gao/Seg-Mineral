"""
Interactive Mask Selection Tool for Image Segmentation

This module provides a high-performance, interactive visualization tool for selecting
and managing image segmentation masks. Optimized for responsive user interaction
and professional workflows.

Author: ZJ Gao
Version: 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass
import json
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Set
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaskSelector:
    """
    High-performance interactive mask selection tool for image segmentation workflows.
    
    Features:
    - Single-click selection with immediate visual feedback
    - Optimized rendering for large datasets
    - Professional error handling and logging
    - Spatial indexing for fast mask lookup
    """
    
    def __init__(self, image_path: str, json_path: str, selection_radius: float = 15.0):
        """
        Initialize the MaskSelector with optimized data structures.
        
        Args:
            image_path: Path to the source image
            json_path: Path to the JSON file containing mask data
            selection_radius: Click detection radius in pixels
        """
        self.image_path = Path(image_path)
        self.json_path = Path(json_path)
        self.selection_radius = selection_radius
        
        # Core data
        self.image = None
        self.masks: List[Dict] = []
        self.mask_centroids: Dict[int, Tuple[float, float]] = {}
        
        # Selection state
        self.selected_indices: Set[int] = set()
        self.visible_indices: Set[int] = set()
        
        # UI components
        self.fig = None
        self.ax = None
        self.text_objects: Dict[int, plt.Text] = {}
        self.highlight_circles: Dict[int, Circle] = {}
        
        # Performance optimization
        self._last_draw_time = 0
        self._draw_throttle = 0.016  # 60 FPS limit
        
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and preprocess image and mask data with error handling."""
        try:
            # Load image
            if not self.image_path.exists():
                raise FileNotFoundError(f"Image file not found: {self.image_path}")
            self.image = plt.imread(self.image_path)
            logger.info(f"Loaded image: {self.image_path} ({self.image.shape})")
            
            # Load masks
            if not self.json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {self.json_path}")
                
            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.masks = json.load(file)
                
            # Precompute centroids for performance
            self._precompute_centroids()
            logger.info(f"Loaded {len(self.masks)} masks with precomputed centroids")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _precompute_centroids(self) -> None:
        """Precompute mask centroids for fast spatial lookup."""
        for idx, mask in enumerate(self.masks):
            try:
                segmentation_mask = np.array(mask['segmentation'], dtype=np.uint8)
                if np.any(segmentation_mask):
                    y, x = center_of_mass(segmentation_mask)
                    self.mask_centroids[idx] = (float(x), float(y))
            except Exception as e:
                logger.warning(f"Failed to compute centroid for mask {idx}: {e}")
    
    def _find_nearest_mask(self, x: float, y: float) -> Optional[int]:
        """
        Fast spatial lookup to find the nearest mask to click coordinates.
        
        Args:
            x, y: Click coordinates
            
        Returns:
            Index of nearest mask within selection radius, or None
        """
        min_distance = float('inf')
        selected_idx = None
        
        for idx in self.visible_indices:
            if idx not in self.mask_centroids:
                continue
                
            mask_x, mask_y = self.mask_centroids[idx]
            distance = np.sqrt((x - mask_x) ** 2 + (y - mask_y) ** 2)
            
            if distance < min_distance and distance <= self.selection_radius:
                min_distance = distance
                selected_idx = idx
                
        return selected_idx
    
    def _throttled_draw(self) -> None:
        """Throttle canvas redraws to maintain performance."""
        current_time = time.time()
        if current_time - self._last_draw_time > self._draw_throttle:
            self.fig.canvas.draw_idle()
            self._last_draw_time = current_time
    
    def _update_mask_visual(self, mask_idx: int) -> None:
        """
        Update visual representation of a single mask.
        
        Args:
            mask_idx: Index of mask to update
        """
        if mask_idx not in self.mask_centroids:
            return
            
        x, y = self.mask_centroids[mask_idx]
        is_selected = mask_idx in self.selected_indices
        
        # Remove existing visuals
        if mask_idx in self.text_objects:
            self.text_objects[mask_idx].remove()
        if mask_idx in self.highlight_circles:
            self.highlight_circles[mask_idx].remove()
        
        # Create new visuals
        if is_selected:
            # Selected state: yellow background with emphasis
            circle = Circle((x, y), 8, color='yellow', alpha=0.8, zorder=10)
            self.ax.add_patch(circle)
            self.highlight_circles[mask_idx] = circle
            
            text = self.ax.text(x, y, str(mask_idx), color='black', fontsize=9, 
                              fontweight='bold', ha='center', va='center', zorder=11)
        else:
            # Default state: clean white background
            text = self.ax.text(x, y, str(mask_idx), color='red', fontsize=8,
                              ha='center', va='center', zorder=10,
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', 
                                       boxstyle='round,pad=0.2'))
        
        self.text_objects[mask_idx] = text
    
    def _handle_click(self, event) -> None:
        """
        Optimized click handler with immediate response.
        
        Args:
            event: Matplotlib mouse event
        """
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
            
        mask_idx = self._find_nearest_mask(event.xdata, event.ydata)
        
        if mask_idx is None:
            return
            
        # Toggle selection state
        if event.button == 1:  # Left click - select
            if mask_idx in self.selected_indices:
                logger.info(f"Mask {mask_idx} already selected")
            else:
                self.selected_indices.add(mask_idx)
                logger.info(f"Selected mask {mask_idx}")
                
        elif event.button == 3:  # Right click - deselect
            if mask_idx in self.selected_indices:
                self.selected_indices.remove(mask_idx)
                logger.info(f"Deselected mask {mask_idx}")
            else:
                logger.info(f"Mask {mask_idx} not in selection")
        
        # Update visual immediately
        self._update_mask_visual(mask_idx)
        self._throttled_draw()
    
    def display_interactive_interface(self, filter_indices: Optional[List[int]] = None) -> List[int]:
        """
        Launch the interactive mask selection interface.
        
        Args:
            filter_indices: List of mask indices to hide from display
            
        Returns:
            List of selected mask indices
        """
        # Determine visible masks
        all_indices = set(range(len(self.masks)))
        hidden_indices = set(filter_indices or [])
        self.visible_indices = all_indices - hidden_indices
        
        logger.info(f"Displaying {len(self.visible_indices)} masks "
                   f"({len(hidden_indices)} filtered out)")
        
        # Calculate figure size based on image dimensions (original scale)
        dpi = 100  # Standard DPI
        img_height, img_width = self.image.shape[:2]
        fig_width = img_width / dpi
        fig_height = img_height / dpi
        
        # Ensure reasonable window size (limit max size for very large images)
        max_size = 15  # Maximum dimension in inches
        if fig_width > max_size or fig_height > max_size:
            scale_factor = max_size / max(fig_width, fig_height)
            fig_width *= scale_factor
            fig_height *= scale_factor
        
        # Create figure with calculated size
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height), 
                                        facecolor='white', dpi=dpi)
        self.fig.suptitle('Interactive Mask Selection Tool', fontsize=14, fontweight='bold')
        
        # Display image at original scale
        self.ax.imshow(self.image, aspect='equal', interpolation='nearest')
        self.ax.set_title('Left-click: Select | Right-click: Deselect', fontsize=10)
        self.ax.axis('off')
        
        # Initialize all visible mask visuals
        for mask_idx in self.visible_indices:
            self._update_mask_visual(mask_idx)
        
        # Connect optimized event handler
        self.fig.canvas.mpl_connect('button_press_event', self._handle_click)
        
        # Display instructions
        self._show_instructions()
        
        # Show interface with proper window management and layout
        plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)  # Add proper margins
        
        # Additional window management
        try:
            # Make sure window is visible and on top
            self.fig.canvas.manager.show()
            if hasattr(self.fig.canvas.manager, 'window'):
                window = self.fig.canvas.manager.window
                if hasattr(window, 'lift'):  # Tkinter
                    window.lift()
                    window.attributes('-topmost', True)
                    window.attributes('-topmost', False)  # Remove topmost after bringing to front
                elif hasattr(window, 'raise_'):  # Qt
                    window.raise_()
                    window.activateWindow()
        except Exception as e:
            logger.warning(f"Could not manage window visibility: {e}")
            
        plt.show()
        
        return sorted(list(self.selected_indices))
    
    def _show_instructions(self) -> None:
        """Display user instructions."""
        instructions = [
            "=== INTERACTIVE MASK SELECTION TOOL ===",
            "• Left-click: Select mask (yellow highlight)",
            "• Right-click: Deselect mask (white highlight)", 
            "• Close window when finished",
            f"• Selection radius: {self.selection_radius} pixels",
            f"• Total masks: {len(self.masks)} | Visible: {len(self.visible_indices)}",
            "=" * 45
        ]
        
        for instruction in instructions:
            print(instruction)
    
    def get_selection_summary(self, base_filter: Optional[List[int]] = None) -> Dict:
        """
        Generate a comprehensive selection summary.
        
        Args:
            base_filter: Base list of filtered masks to combine with selection
            
        Returns:
            Dictionary containing selection statistics and combined results
        """
        base_filter = base_filter or []
        selected_list = sorted(list(self.selected_indices))
        combined_removal = sorted(set(base_filter + selected_list))
        
        return {
            'newly_selected': selected_list,
            'base_filter': base_filter,
            'total_removal_list': combined_removal,
            'selection_count': len(selected_list),
            'total_removal_count': len(combined_removal),
            'total_masks': len(self.masks)
        }


def main():
    """Example usage demonstrating the professional workflow."""
    
    # Configuration
    image_path = 'Aligned/USU-4183B 150-250 Elemental Map/USU-4183B 150-250 Elemental Map_aligned.png'
    json_path = 'Seg_Images/USU-4183B 150-250 Elemental Map_aligned/USU-4183B 150-250 Elemental Map_aligned.json'
    
    # Pre-existing filter list
    base_filter = [82, 86, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 114, 115]
    
    try:
        # Initialize selector
        selector = MaskSelector(image_path, json_path, selection_radius=15.0)
        
        # Launch interactive selection
        selected_masks = selector.display_interactive_interface(filter_indices=base_filter)
        
        # Generate comprehensive summary
        summary = selector.get_selection_summary(base_filter)
        
        # Professional output
        print("\n" + "=" * 60)
        print("MASK SELECTION SUMMARY")
        print("=" * 60)
        print(f"Newly selected masks: {summary['newly_selected']}")
        print(f"Selection count: {summary['selection_count']}")
        print(f"Combined removal list: {summary['total_removal_list']}")
        print(f"Total removal count: {summary['total_removal_count']}")
        print(f"Remaining masks: {summary['total_masks'] - summary['total_removal_count']}")
        print("=" * 60)
        
        # Copy-ready output
        print("\nCOPY-READY REMOVAL LIST:")
        print("-" * 30)
        print(summary['total_removal_list'])
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()