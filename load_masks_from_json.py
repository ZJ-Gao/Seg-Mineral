import json
import numpy as np


def load_masks_from_json(json_path):
    """
    Load masks from a JSON file.

    Parameters:
    - json_path (str): The path to the JSON file.

    Returns:
    - masks: The loaded masks.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        masks = json.load(file)
    
    # Convert segmentation lists back to numpy arrays
    for mask in masks:
        mask['segmentation'] = np.array(mask['segmentation'])
    
    return masks