# Homework: Image Compression using K-Means Clustering

# In this assignment, you will apply the K-means clustering algorithm for image compression.
# The goal is to reduce the number of colors in an image by clustering similar colors together.
# This is a practical application of K-means in the field of image processing and computer vision.

# Implement the K-means clustering part of the image compression function. You are provided
# with a template where the image loading and preparation steps are done.
# Your task is to complete the compress_image function by implementing the K-means algorithm
# to find the main color clusters and then map each pixel of the image to the nearest cluster centroid.


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image


def compress_image1(image_path, num_colors):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Get the number of pixels in the image
    num_pixels = img_np.shape[0] * img_np.shape[1]

    # Reshape the image to be a list of pixels
    pixels = img_np.reshape(num_pixels, 3)

    # Apply K-means clustering to find cluster centroids (the new colors)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)

    # Replace pixel values with the nearest centroids
    compressed_pixels = kmeans.predict(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)

    # Invert colors of cluster centroids
    inverted_cluster_centers = 255 - cluster_centers

    # Create a new image using the inverted cluster centers
    inverted_compressed_image = inverted_cluster_centers[compressed_pixels].reshape(img_np.shape)

    # Convert the numpy array back to an image
    inverted_compressed_image = Image.fromarray(np.uint8(inverted_compressed_image), 'RGB')

    return inverted_compressed_image


def compress_image(image_path, num_colors):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Reshape the image to be a list of pixels
    pixels = img_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=12).fit(pixels)
    predicted_pixels = kmeans.predict(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    new_image = cluster_centers[predicted_pixels].reshape(img_np.shape)

    # This line ensures the data type is correct and values are within the valid range
    new_pixels = np.clip(new_image.astype('uint8'), 0, 255)

    # Reshape the new_pixels array to the original image's dimensions
    new_img_np = new_pixels.reshape(img_np.shape)

    # Convert back to an image
    new_img = Image.fromarray(new_img_np)
    return new_img


# Example usage
image_path = 'Rickroll.jpg'  # Replace with your image path
compressed_image = compress_image(image_path, num_colors=16)
plt.imshow(compressed_image)
plt.show()

