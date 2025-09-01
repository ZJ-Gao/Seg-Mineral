import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
    This code will return a mask tie to the grain you click. Each time after you click.
    Remember to copy the index list you've decided to move and paste to `filter_list`


"""

class MaskSelector:
    """
    A class for selecting masks interactively from an image and displaying corresponding masks.
    """

    def __init__(self, image_path, json_path):
        self.image_path = image_path
        self.masks = self.load_masks_from_json(json_path)
        self.coords = []
        self.indexes = []
        self.index_of_interest_list = []

    def load_masks_from_json(self, json_path):
        """
        Load masks from a JSON file and convert them to numpy arrays.
        """
        with open(json_path, 'r', encoding='utf-8') as file:
            masks = json.load(file)
        for mask in masks:
            mask['segmentation'] = np.array(mask['segmentation'])
        return masks

    def onclick(self, event, filter_list=None):
        """
        Handle mouse click events. Registers clicked points and matches them to mask indexes.
        """
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if x is not None and y is not None:
                self.coords.append((x, y))
                print(f'Coordinates: {self.coords[-1]}')
                idx = self.match_mask_to_coords(x, y, filter_list)
                if idx is not None:
                    self.indexes.append(idx)
                    self.index_of_interest_list.append(idx)
                    event.inaxes.plot(x, y, 'ro')
                    event.canvas.draw()

    def match_mask_to_coords(self, x, y, filter_list=None):
        """
        Match the clicked coordinates to the corresponding mask index.
        """
        for idx, mask in enumerate(self.masks):
            if filter_list is not None and idx in filter_list:
                continue
            if mask['segmentation'][y, x]:
                print(f"Mask index: {idx}")
                return idx
        return None

    def interactive_mask_selection(self, filter_list=None):
        """
        Set up the interactive mask selection process with the given filter list.
        """
        self.coords = []
        self.indexes = []
        self.index_of_interest_list = []

        image = Image.open(self.image_path)
        image_np = np.array(image)

        fig, ax = plt.subplots()
        ax.imshow(image_np)
        fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event, filter_list))
        plt.show()

        return self.index_of_interest_list

    def display_final_result(self):
        """
        Display the final image with all selected points, indexes, and overlaid masks after all clicks.
        """
        # Load the original image
        image = Image.open(self.image_path)
        image_np = np.array(image)

        # Create an overlay for all selected masks
        mask_overlay = np.zeros_like(image_np)

        for idx in self.indexes:
            mask = self.masks[idx]['segmentation']
            mask_overlay[mask > 0] = [255, 0, 0]  # Red for mask

        # Display the image with the mask overlay and click points
        fig, ax = plt.subplots()
        ax.imshow(image_np, alpha=0.7)
        ax.imshow(mask_overlay, alpha=0.3)  # Overlay all masks with some transparency

        # Mark each clicked point and show its corresponding index
        for (x, y), idx in zip(self.coords, self.indexes):
            ax.plot(x, y, 'ro')  # Red dot for each click
            ax.text(x, y, str(idx), color='white', fontsize=12, ha='center', va='center')

        ax.set_title(f"Clicked Masks: {self.indexes}")
        plt.show()

# Example usage
if __name__ == "__main__":
    image_path = 'To_be_seg/4536 Elemental Map.tiff'
    json_path = 'Seg_Images/4536 Elemental Map/4536 Elemental Map.json'
    # filter_list_USU_4183B_250_355 = [109, 92, 88, 99, 26, 107, 81, 87, 103, 76, 97, 100, 55, 93, 37, 110, 68, 91, 106, 96, 108, 86, 84, 40, 98, 83]
    filter_list = [72, 156, 161, 169, 183, 205]
    selector = MaskSelector(image_path, json_path)
    index_of_interest_list = selector.interactive_mask_selection(filter_list=filter_list)
    print("Index of interest list:", index_of_interest_list)

    # Display the final image with all selected masks, points, and indexes
    selector.display_final_result()

