import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Function to get user input for feature values
def get_user_input(features):
    user_input = []
    print("Please enter the values for the following features:")
    for i, feature_name in enumerate(features):
        value = float(input(f"{i+1}. {feature_name}: "))
        user_input.append(value)
    return np.array(user_input).reshape(1, -1)

# Load breast cancer dataset from scikit-learn
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Define callbacks for training
checkpoint_path = "best_model.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch % 3 == 0 else lr)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# Train the model on the training data
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
                    callbacks=[model_checkpoint, lr_scheduler, early_stopping, tensorboard])
print("Training complete!")

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Load the best model from checkpoint
best_model = tf.keras.models.load_model(checkpoint_path)

# Continue until the user is done selecting features

# Get user input for features
user_input = get_user_input(feature_names)

# Make a prediction on the user input using the best model
prediction = best_model.predict(user_input)

# Display the prediction result
if prediction[0][0] >= 0.5:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")

while True:
    # Get the selected feature index from the user
    selected_feature_idx = int(input("Enter the index of the feature you want to visualize (1-30) or enter 0 to exit: ")) - 1

    if selected_feature_idx == -1:
        break

    # Generate random feature scales for visualization
    num_samples = 100
    random_input = np.random.randn(num_samples, 30) * 2  # Random features with higher scales
    scaled_random_input = scaler.transform(random_input)

    # Make predictions on the scaled random features
    random_predictions = best_model.predict(scaled_random_input)

    # Plot the scatter plot for visualization of the selected feature
    plt.figure(figsize=(8, 6))
    plt.scatter(scaled_random_input[:, selected_feature_idx], random_predictions, c=random_predictions, cmap='coolwarm', marker='o', alpha=0.7)
    plt.scatter(user_input[0, selected_feature_idx], prediction, c='red', marker='o', s=100)
    plt.xlabel(feature_names[selected_feature_idx])
    plt.ylabel('Probability')
    plt.title(f'Probabilities of Malignant Predictions for {feature_names[selected_feature_idx]}')
    plt.show()

# Visualize the model using TensorBoard
print("To visualize the model using TensorBoard, run the following command in your terminal:")
print("tensorboard --logdir=logs")
