
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 model without the top (fully connected) layer
pretrained_base = VGG16(weights='imagenet', include_top=False)
import tensorflow as tf
import matplotlib.pyplot as plt

# Assuming the model is defined as per your earlier code
# Load VGG16
# pretrained_base = tf.keras.models.load_model(
#     '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
# )

model = tf.keras.Sequential([
    pretrained_base,
    layers.GlobalAvgPool2D(),
])

# Load dataset (with 5 categories)
dataset_path = 'C:/Users/ldomi/.keras/datasets/flower_photos'  # Update this path if needed
ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',  # Multi-class, use 'int' for integer labels or 'categorical' for one-hot encoded labels
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=1,
    shuffle=True,
)

# Get an example image and label
ds_iter = iter(ds)
example = next(ds_iter)

# Resize and preprocess the image
car_tf = tf.image.resize(example[0], size=[128, 128])

# Extract features using the pre-trained model
car_features = model(car_tf)

# Reshape the features (you can adjust the shape depending on your requirements)
car_features = tf.reshape(car_features, shape=(16, 32))  # Update this as per your feature map size

# Get the label
label = int(tf.squeeze(example[1]).numpy())  # Get the integer label

# Plot the image and its features
plt.figure(figsize=(8, 4))

# Original image
plt.subplot(121)
plt.imshow(tf.squeeze(example[0]))
plt.axis('off')
# Display class name
class_names = ds.class_names  # Class names (e.g., ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
plt.title(class_names[label])

# Feature map
plt.subplot(122)
plt.imshow(car_features)
plt.title('Pooled Feature Maps')
plt.axis('off')

plt.show()

