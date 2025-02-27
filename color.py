import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def closest_color(requested_color):
    """
    Find the closest color name for an RGB tuple from a predefined set of 8 colors.
    """
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "blue": (0, 0, 255),
        "brown": (165, 42, 42),
        "pink": (255, 192, 203)
    }
    
    min_distance = float("inf")
    closest_name = None
    for name, rgb in colors.items():
        distance = sum((requested_color[i] - rgb[i]) ** 2 for i in range(3))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    
    return closest_name

def get_dominant_color(image, k=1):
    """
    Takes an image frame and returns the dominant color name.
    
    :param image: Image frame (numpy array)
    :param k: Number of dominant colors to find (default is 1)
    :return: Dominant color name
    """
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Use KMeans clustering to find dominant color
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the most common color
    dominant_color = tuple(kmeans.cluster_centers_[0].astype(int))
    
    return closest_color(dominant_color)

# Example usage
# dominant_color_name = get_dominant_color(frame)
# print(dominant_color_name)