import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import json
from PIL import Image
import cv2

import matplotlib
matplotlib.use('TkAgg')  # Ensure Tkinter backend

class MaskCleanner:
    """
    A class to visualize masks on images and interactively select masks for removal.
    """
    def __init__(self, image_path, json_path):
        self.image = plt.imread(image_path)
        self.masks = self.load_masks_from_json(json_path)
        self.indices_to_remove = []

    def load_masks_from_json(self, json_path):
        # Load masks from a JSON file and convert segmentation lists to numpy arrays.
        with open(json_path, 'r', encoding='utf-8') as file:
            masks = json.load(file)
        for mask in masks:
            mask['segmentation'] = np.array(mask['segmentation'])
        return masks

    def onclick(self, event, fig, ax):
        # Event handler for mouse click, to select or deselect masks based on proximity to the click.
        min_distance = float('inf')
        selected_idx = None
        selected_coords = None

        for idx, mask in enumerate(self.masks):
            segmentation_mask = mask['segmentation'].astype(np.uint8)
            if np.any(segmentation_mask):
                y, x = center_of_mass(segmentation_mask)
                distance = np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)
                if distance < min_distance and distance < 10:
                    min_distance = distance
                    selected_idx = idx
                    selected_coords = (x, y)

        if selected_idx is not None:
            self.handle_selection(event, selected_idx, selected_coords, fig, ax)

    def handle_selection(self, event, idx, coords, fig, ax):
        # Handle mask selection or deselection based on mouse click event.
        x, y = coords
        if event.button == 1:  # Left click to select
            if idx not in self.indices_to_remove:
                self.indices_to_remove.append(idx)
                print(f"Mask {idx} added to remove list.")
                ax.text(x, y, f"{idx}", color='blue', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))
            else:
                print(f"Mask {idx} is already in the remove list.")
        elif event.button == 3:  # Right click to deselect
            if idx in self.indices_to_remove:
                self.indices_to_remove.remove(idx)
                print(f"Mask {idx} removed from remove list.")
                ax.text(x, y, f"{idx}", color='red', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            else:
                print(f"Mask {idx} is not in the remove list.")
        fig.canvas.draw()

    def display_interactive_labeled_image(self, filter_list=None):
        # Display the image with labeled masks and allow for interactive mask selection/removal.
        idx_list = list(range(len(self.masks)))
        if filter_list:
            idx_list = [idx for idx in idx_list if idx not in filter_list]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image)
        ax.axis('off')

        for idx, segmentation_entry in enumerate(self.masks):
            if idx in idx_list:
                segmentation_mask = np.array(segmentation_entry['segmentation']).astype(np.uint8)
                if np.any(segmentation_mask):
                    y, x = center_of_mass(segmentation_mask)
                    ax.text(x, y, str(idx), color='red', fontsize=8, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event, fig, ax))

        fig.show()  # Add this line to ensure the window is displayed
        plt.show()

        return self.indices_to_remove


# Example usage:
if __name__ == "__main__":
    image_path = 'To_be_seg/4536 Elemental Map.tiff'

    json_path = 'Seg_Images/4536 Elemental Map/4536 Elemental Map.json'

    
    visualizer = MaskCleanner(image_path, json_path)

    filter_list = [72, 163, 166, 168, 169, 181, 191, 194, 195, 196, 197, 200, 201, 202, 204, 205, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 227, 228]
    
    # Display the image with masks and interactively select masks to remove
    clicked_indices = visualizer.display_interactive_labeled_image(filter_list=filter_list)

    # Combine the filter list and clicked indices
    final_labels_to_remove = sorted(set(filter_list + clicked_indices))

    print("Manaul clicked labels this time:", clicked_indices)
    print("Combined list of masks to be removed:", final_labels_to_remove)
    print("--------------------------------------------------")
    print("Copy this: Combined list of masks to be removed")
    print("--------------------------------------------------")
