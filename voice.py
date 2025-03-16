# Import necessary libraries
from datasets import load_dataset
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Step 1: Load the dataset from Hugging Face
dataset = load_dataset("cats_vs_dogs")

# Step 2: Define preprocessing parameters
IMG_SIZE = (224, 224)  # Size expected by MobileNetV2
BATCH_SIZE = 32

# Preprocessing function for each example
def preprocess(example):
    image = example['image']
    label = example['labels']
    
    # Resize and normalize the image
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Convert label to one-hot encoding (2 classes: cat and dog)
    label = tf.one_hot(label, depth=2)
    return image, label

# Apply preprocessing to the training split
train_dataset = dataset['train'].map(preprocess, remove_columns=['image', 'labels'])

# Convert to tf.data.Dataset for efficient training
train_tf_dataset = tf.data.Dataset.from_tensor_slices((
    [example['image'] for example in train_dataset],
    [example['labels'] for example in train_dataset]
)).batch(BATCH_SIZE)

# Step 3: Load and configure the model
# Use pre-trained MobileNetV2 from TensorFlow Hub
base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
                            trainable=False)  # Freeze base layers

# Build the model with a classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: cat and dog
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
model.fit(train_tf_dataset, epochs=3)

# Step 5: Save the trained model
model.save('cats_vs_dogs_model.h5')

print("Model training complete and saved as 'cats_vs_dogs_model.h5'")
