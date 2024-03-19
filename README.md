\# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>
Import numpy module as np and pandas as pd.

### Step2:
<br>
Assign the values to variables in the program.

### Step3:
<br>
Get the values from the user appropriately.

### Step4:
<br>
Continue the program by implementing the codes of required topics.

### Step5:
<br>
Thus the program is executed in google colab.

## Program:
```python
Developed By:MOHAMMED IMTHIYAS.M
Register Number:212222230083
```
i)Image Translation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("cat.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()
```

ii) Image Scaling
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)
```


iii)Image shearing
```import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '3nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```

iv)Image Reflection
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '4 nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```

v)Image Rotation
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nat5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)


```



vi)Image Cropping
```

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '6nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)




```
## Output:
### i)Image Translation

![312325094-548046b8-6518-42c6-9a64-189c5e04f929](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/766b3af5-4c90-4cc8-a054-29c026a6e53c)


<br>
<br>

### ii) Image Scaling
<br>

![312325109-29fa214d-3ccd-4d17-9b32-1867c176ea93](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/c73d01b5-1b2f-4d7d-a0e3-08896e2b5bef)





<br>
<br>


### iii)Image shearing
<br>



![312325133-c5d288e3-b44a-4b9b-96dd-68f56cc0bc98](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/e8183f5e-7131-4a68-a976-14a538bf1cb9)







<br>
<br>


### iv)Image Reflection
<br>





![312325162-41636717-c04c-4afe-a511-3c1f098f454c](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/f0ddc5a4-b160-42af-8a50-8bdb4c8ca451)





<br>
<br>



### v)Image Rotation
<br>


![312325194-a66e4df3-1b1a-4dbe-85a1-2bd429f6369f](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/695e491b-b8e3-42ef-9abe-d0255ebfe45f)







<br>
<br>



### vi)Image Cropping
<br>












![312325214-e5967a8e-fe45-4193-baa7-61240c7719be](https://github.com/imthiyas19/IMAGE-TRANSFORMATIONS/assets/120353416/104b4957-e84c-424a-9f56-2863a714ac02)

<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
