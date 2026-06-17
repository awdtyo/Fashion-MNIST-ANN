# Fashion-MNIST Image Classification

A fully-connected neural network for classifying Fashion-MNIST images, built with TensorFlow/Keras. The model trains with early stopping on validation loss and reaches **88.04% test accuracy** in 6 epochs.

## Overview

Fashion-MNIST is a drop-in replacement for the classic MNIST dataset: 70,000 grayscale 28×28 images spanning 10 clothing categories. This project covers the full workflow — loading and normalizing the data, defining a multi-layer perceptron (MLP), training with early stopping, evaluating on the test set, and visualizing the resulting loss curves.

## Dataset

- **Source:** `tensorflow.keras.datasets.fashion_mnist`
- **Training set:** 60,000 images, 28×28 grayscale
- **Test set:** 10,000 images, 28×28 grayscale
- **Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Preprocessing:** pixel values scaled from `[0, 255]` to `[0, 1]` by dividing by 255

## Model Architecture

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

| Layer | Output Shape | Parameters | Activation |
|---|---|---|---|
| Flatten | 784 | 0 | – |
| Dense | 128 | 100,480 | ReLU |
| Dense | 64 | 8,256 | ReLU |
| Dense (output) | 10 | 650 | Softmax |

**Total trainable parameters:** ~109,386

A plain MLP baseline — no convolutional layers, so it doesn't exploit spatial structure in the images.

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Max epochs | 100 |
| Early stopping | `monitor='val_loss'`, `patience=2`, `restore_best_weights=True` |
| Validation data | Test split (`X_test`, `y_test`) |

Training ran for 6 epochs before early stopping triggered (2 consecutive epochs without validation loss improvement). Final weights were restored from **epoch 4**, the point of lowest validation loss.

## Results

| Metric | Value |
|---|---|
| Test loss | 0.3367 |
| Test accuracy | 88.04% |



Training loss decreases steadily throughout, while validation loss bottoms out around epoch 4 before drifting upward. Early stopping correctly halted training near the point of best generalization rather than letting the model continue overfitting through the full 100-epoch budget.

## Project Structure

```
.
├── Fashion_MNIST.ipynb     # Main notebook: data loading, model, training, evaluation
├── model.h5                # Saved trained model (HDF5 format)
├── images/
│   └── loss_curve.png      # Training/validation loss plot
└── README.md
```

## Requirements

```
tensorflow
numpy
matplotlib
```

## Usage

1. Open `Fashion_MNIST.ipynb` in Jupyter or Google Colab (originally run on a Colab T4 GPU runtime).
2. Run all cells top to bottom. Fashion-MNIST downloads automatically via `tensorflow.keras.datasets.fashion_mnist`.
3. The trained model is saved to `model.h5`. To reload it later:

```python
from tensorflow.keras.models import load_model
model = load_model('model.h5')
```

## Notes & Possible Improvements

- `model.save('model.h5')` uses the legacy HDF5 format; Keras now recommends the native format, e.g. `model.save('model.keras')`.
- The architecture is a plain MLP rather than a CNN. A convolutional model would likely push past the current ~88% accuracy ceiling by exploiting spatial patterns in the images.
- `patience=2` is fairly aggressive for a validation curve with noisy fluctuations. Trying a higher patience (3–4) is worth testing to see whether it surfaces a better minimum before stopping.
- No data augmentation or dropout/regularization is used — both are natural next steps if extending this baseline.

