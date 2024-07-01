# EgoHands Data Preprocessing and Visualization

![](https://source.roboflow.com/5w20VzQObTXjJhTjq6kad9ubrm33/9ExkEYuMdWqL6NN63SBY/original.jpg)

This project demonstrates various image preprocessing and visualization techniques applied to the EgoHands dataset. 
The steps include downloading the dataset, reading images, applying grayscale transformation, histogram equalization, edge detection, and superpixel segmentation.

## Requirements

- Python 
- NumPy
- Matplotlib
- scikit-image
- requests
- scipy

## Steps

### 1. Download and Organize the Dataset

The dataset is downloaded from the EgoHands website, extracted, and organized into a directory.

```python
import os
import requests

url = 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip'
r = requests.get(url, allow_redirects=True)
open('egohands_data.zip', 'wb').write(r.content)

!rm -r data
!rm -r temp
!mkdir data
!mkdir temp
!unzip egohands_data.zip -d temp/
!cp -r /content/temp/_LABELLED_SAMPLES/CARDS_OFFICE_B_S/* /content/data/
!rm /content/data/polygons.mat
!rm -r temp
```

### 2. Read Images from the Directory

Images are read from the 'data' directory into a list.

```python
from skimage import io

dataset_dir = "data" 
imgs = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(dataset_dir, filename)
        img = io.imread(img_path)
        imgs.append(img)

print("Number of images read:", len(imgs))
```

### 3. Draw Sampled Images in a Grid

A function to draw 9 sampled images from the list in a 3x3 grid using Matplotlib.

```python
import matplotlib.pyplot as plt
import random

random.seed(100)

def draw_func(list_of_imgs):
    sampled_imgs = random.sample(list_of_imgs, 9)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, img in enumerate(sampled_imgs):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

    plt.show()

draw_func(imgs)
```

### 4. Apply Grayscale Transformation

Convert the images to grayscale.

```python
from skimage import color

gray_scale_imgs = [color.rgb2gray(img) for img in imgs]
draw_func(gray_scale_imgs)
```

### 5. Apply Histogram Equalization

Enhance the contrast of the grayscale images.

```python
from skimage import exposure

equ_imgs = [exposure.equalize_hist(img) for img in gray_scale_imgs]
draw_func(equ_imgs)
```

### 6. Apply Sobel Edge Detection

Detect edges in the equalized images using the Sobel filter.

```python
from skimage.filters import sobel

sobel_imgs = [sobel(img) for img in equ_imgs]
draw_func(sobel_imgs)
```

### 7. Apply Gaussian Derivative

Compute the Gaussian derivative for edge detection.

```python
import scipy.ndimage
import numpy as np

weights = np.zeros((9, 9))
weights[4, 4] = 1.0
gaussF = scipy.ndimage.filters.gaussian_filter(weights, 1.5, order=0, truncate=3.0)

weights = np.zeros((3,3))
weights[:,0] = -np.ones((3,))
weights[:,2] = np.ones((3,))
DoGx = scipy.ndimage.convolve(gaussF, weights)

weights = np.zeros((3,3))
weights[0,:] = np.ones((3,))
weights[2,:] = -np.ones((3,))
DoGy = scipy.ndimage.convolve(gaussF, weights)

devX = [scipy.ndimage.convolve(img, DoGx) for img in equ_imgs]
devY = [scipy.ndimage.convolve(img, DoGy) for img in equ_imgs]

mag_imgs = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(devX, devY)]
angle = [np.arctan2(dy, dx) for dx, dy in zip(devX, devY)]

draw_func(mag_imgs)
draw_func(angle)
```

### 8. Apply Superpixel Segmentation

Segment the images into superpixels.

```python
from skimage.segmentation import slic

super_pixels = [slic(img, n_segments=100, compactness=10) for img in equ_imgs]
draw_func(super_pixels)
```

## Conclusion

This notebook provides a comprehensive workflow for image preprocessing and visualization, showcasing several fundamental image processing techniques.
