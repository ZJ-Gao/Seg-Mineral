import os

class FolderImagePath:
    def __init__(self, folder_path, keyword='K', extensions=None):
        """
        Initializes the FolderImagePath class.
        
        Parameters:
        - folder_path: Path to the folder containing images.
        - keyword: Keyword to decide the second image in the list (default is 'Si').
        - extensions: Set of acceptable image file extensions. Defaults to common formats.
        """
        self.folder_path = folder_path
        self.keyword = keyword
        self.extensions = extensions if extensions else {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    def get_images_path(self):
        """
        Returns a list of paths to image files in the specified folder.
        
        Returns:
        - List of full paths to image files in the folder.
        """
        return [
            os.path.join(self.folder_path, filename)
            for filename in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, filename)) and
               os.path.splitext(filename)[1].lower() in self.extensions
        ]

    def generate_image_path_list(self):
        """
        Generates a list of image paths in a specific order:
        1. 'Elemental' image first (if it exists),
        2. Image with the specified keyword second (if it exists),
        3. All remaining images afterward.
        
        Returns:
        - Ordered list of image paths based on the specified conditions.
        """
        # Retrieve all image paths with valid extensions
        image_paths = self.get_images_path()
        
        # Initialize placeholders for specific images
        elemental_image = None
        keyword_image = None
        other_images = []

        # Classify images based on filenames
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            
            # Check for "Elemental" in the filename
            if "Elemental" in filename:
                elemental_image = image_path
            # Check for keyword in the filename
            elif self.keyword in filename:
                keyword_image = image_path
            # Add to others if it doesn't match the above criteria
            else:
                other_images.append(image_path)

        # Build the final ordered list of image paths
        ordered_image_paths = []
        if elemental_image:
            ordered_image_paths.append(elemental_image)
        if keyword_image:
            ordered_image_paths.append(keyword_image)
        ordered_image_paths.extend(other_images)

        # Output the ordered image path list for debugging
        print("Generated Ordered Image Path List:")
        for path in ordered_image_paths:
            print(path)

        return ordered_image_paths
