This project is an integrate pipeline for extracting mineral grains from SEM images and then use it to train a deep learning model for mineral grain classification.

The progress has been made in the following steps:
1. Align images with the same field of view but taken under different elements being illuminated
2. Use segment-anything model to segment mineral grains
3. Visualize the segmented result and evaluate the following:
- Have all the grains been segmented? Any grains being left out? Any grains got segmented more than once or segmented incorrectly with other grains together?
- Are the segmented grains complete? Any grains being cut off?
Another visualization is to display different elementary images of the same grain together to observe what's the dominant element.
4. Clean the segmented result
5. A little tool to manually pick out grains of interest and calculate percentage in terms of pixel area
6. An interactive script to label the grains with a predefined mineral list, and prepare the dataset in the format that can be directly used for training a deep learning model. Please see a detailed readme here: [Interactive_Labeling_ReadMe](Interactive_Labeling_ReadMe.md)
7. Train a deep learning model for mineral grain classification using fast.ai (work in progress)