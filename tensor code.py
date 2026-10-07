import tensorflow as tf
import time

# 1. Load and normalize the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build the standard CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 3. Start the clock
print("Training started...")
start_time = time.time()

# 4. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 5. Stop the clock and capture GPU memory metrics
end_time = time.time()
total_time = end_time - start_time

# Fetch peak GPU memory usage from Google Colab and convert from Bytes to Megabytes
memory_info = tf.config.experimental.get_memory_info('GPU:0')
peak_memory_mb = memory_info['peak'] / (1024 * 1024)

print("\n--- TensorFlow Benchmark Results ---")
print(f"Total Training Time: {total_time:.2f} seconds")
print(f"Peak GPU Memory Allocated: {peak_memory_mb:.2f} MB")