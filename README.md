# 📝 Text Classification with Deep Learning

A collection of NLP text classification projects built using TensorFlow and TensorFlow Hub, progressing from binary sentiment analysis to multi-class question categorization.

---

## 📁 Projects

### 1. IMDB Movie Review Sentiment Analysis
Binary classification of movie reviews as **positive** or **negative**.
- Dataset: 50,000 IMDB movie reviews
- Accuracy: **85%**
- Type: Binary Classification

### 2. Stack Overflow Question Classifier
Multi-class classification of developer questions into 4 categories.
- Dataset: 16,000+ Stack Overflow questions
- Categories: **Python · JavaScript · Java · C#**
- Type: Multi-class Classification

---

## 🗂️ Project Structure

```
project/
│
├── stack_overflow_16k/
│   ├── train/
│   │   ├── python/
│   │   ├── javascript/
│   │   ├── java/
│   │   └── csharp/
│   └── test/
│
├── logs/                        # TensorBoard logs
├── models/                      # Saved models
├── imdb_classification.ipynb    # IMDB project notebook
└── stackoverflow_classification.ipynb  # Stack Overflow project notebook
```

---

## 🧠 Model Architecture

```
Input (text string)
        ↓
Google NNLM Embedding Layer (TensorFlow Hub)
        ↓
Dense Layer (64 units, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense Layer (32 units, ReLU)
        ↓
Output Dense Layer (softmax)
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.12
- CUDA-compatible GPU (tested on RTX 4060 8GB)
- WSL2 (for Windows users)

### Install dependencies

```bash
# Create virtual environment
python -m venv ml
source ml/bin/activate  # Linux/WSL2

# Install packages
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install tensorflow-hub
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Environment setup (important)

Add this to your `~/.bashrc` to fix Keras compatibility:

```bash
export TF_USE_LEGACY_KERAS=1
```

Or at the top of every notebook before imports:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must be before TF import

import tensorflow as tf
import tensorflow_hub as hub
```

---

## 🚀 Usage

### Load and preprocess data

```python
raw_train_data = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/train",
    batch_size=32,
    subset="training",
    seed=42,
    validation_split=0.2
)

raw_val_data = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/train",
    batch_size=32,
    subset="validation",
    seed=42,
    validation_split=0.2
)
```

### Build model

```python
url = "https://tfhub.dev/google/nnlm-en-dim128/2"
hub_layer = hub.KerasLayer(url, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")  # 4 for Stack Overflow, 1 for IMDB
])

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
```

### Train

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

model.fit(
    raw_train_data,
    validation_data=raw_val_data,
    epochs=20,
    callbacks=[early_stopping]
)
```

---

## 📊 Results

| Project | Dataset Size | Classes | Accuracy |
|---|---|---|---|
| IMDB Sentiment | 50,000 reviews | 2 (pos/neg) | **85%** |
| Stack Overflow | 16,000 questions | 4 (languages) | In progress |

---

## 🔧 GPU Setup (WSL2)

```bash
# Upgrade to WSL2
wsl --update
wsl --set-default-version 2

# Enable memory growth to prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Free VRAM between runs

```python
import tensorflow as tf
import gc

tf.keras.backend.clear_session()
gc.collect()
```

---

## 📈 TensorBoard

```python
import datetime

def create_tensorboard_callback(logdir="logs"):
    log_path = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_path)
```

```bash
# Launch TensorBoard
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## 🐛 Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `hub.KerasLayer` not instance of `keras.Layer` | Keras 3 incompatibility | Use `tensorflow==2.15.0` or set `TF_USE_LEGACY_KERAS=1` |
| `ResourceExhaustedError` OOM | GPU VRAM full | Reduce batch size or run `clear_session()` |
| `ValueError: Empty logs` | Dataset exhausted before training | Recreate dataset or add `.cache()` |
| `from_logits` warning | Softmax + `from_logits=True` mismatch | Use `from_logits=False` with softmax |

---

## 📦 Tech Stack

- **TensorFlow 2.15** — Deep learning framework
- **Keras 2.15** — Model building API
- **TensorFlow Hub** — Pretrained embeddings (NNLM)
- **WSL2** — Linux environment on Windows
- **RTX 4060 8GB** — GPU training
- **Jupyter Notebook** — Development environment

---

## 📚 References

- [TensorFlow Text Classification Tutorial](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
- [TensorFlow Hub NNLM](https://tfhub.dev/google/nnlm-en-dim128/2)
- [Stack Overflow Dataset](https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz)
