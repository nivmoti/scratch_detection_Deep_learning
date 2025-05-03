# Wafer Scratch Detection - Deep Learning Solution

This project implements a deep learning pipeline to detect scratches in semiconductor wafers using a custom U-Net model with fixed coordinate image generation and flexible training/prediction flows.

The model class is designed to be simple to use, encapsulated in a single Python class named `Model`, located in `Model/model.py`.

---

## 🔧 How to Use

### 1. **Import the Model Class**

```python
from Model.model import Model
```

### 2. **Initialize the Model with Training Data**

```python
model = Model(df_train=df_wafers)
```

### 3. **Generate Image Data from Wafer Map**

```python
model.makeData()  # Converts tabular data into 256x256 fixed-coordinate .pt images
```

### 4. **Train the Model**

```python
model.train(epoch=10)
```

### 5. **Make Predictions**

```python
IsScratchDie = model.predict(df_wafers_test, yield_threshold=0.7)
df_wafers_test['IsScratchDie'] = IsScratchDie
```

### 6. **Evaluate Results**

```python
model.evaluate_die_level(df_wafers['IsScratchDie'].values, IsScratchDie)
```

---

## 📁 Folder Structure

```
project_root/
├── Model/
│   ├── model.py           # Model class with training and inference logic
│   ├── SaveModel/         # Stores best_model.pth, optimizer, val loss
├── datamap/
│   ├── Waferclass.py      # Dataset and caching logic
│   └── data/cache/        # Cached training images and labels
│   └── data/test/         # Cached test images
```

---

##  Key Features

###  Fixed Coordinate Image Grid

* Each wafer is converted to a 256x256 tensor.
* Coordinates are mapped directly based on `DieX`, `DieY` offset.
* This improves stability and removes ambiguity when mapping predictions back to wafer-level CSV.

###  Training Loop

* Saves best model, loss, and optimizer states.
* Early stopping with patience.
* Automatically resumes training if `continue_training=True`.

###  Wafer Filtering

* Removes low-yield wafers (e.g., yield < 0.7) before predicting.

### ✅Batch Prediction Flow

* Loads preprocessed images.
* Uses the trained model to predict per-die scratch detection.
* Maps prediction back to CSV coordinates.

---

## 📊 Metrics

Model evaluation is performed at the **die level**:

* Accuracy
* Precision
* Recall
* F1-score

Results are printed and also saved to a CSV:

```python
model.evaluate_die_level(true_labels, pred_labels, save_path="metrics.csv")
```

---

## 🧪 Experiments & Observations

### 🔁 Input Format Iterations

* **Original attempts** used stretched resized 256x256 images.
* This introduced distortions during back-conversion to die coordinates.
* 🔁 Final solution uses **fixed coordinate grid**: a pixel is drawn **only if a die exists** in that coordinate.

### 🔁 Prediction Mapping

* Prediction results in a \[2, 256, 256] tensor (2 channels: \[prob\_not\_scratch, prob\_scratch]).
* Argmax is applied along channel dim to extract final binary scratch mask.

### 🔁 Model Improvements

* CrossEntropyLoss with weighting improves imbalance.
* Batch size and DataLoader optimizations improve GPU utilization.
* Patience-based early stopping prevents overfitting.



---

## 📥 Output Example

![image](https://github.com/user-attachments/assets/08bedc2d-9a27-42a1-b2b1-df31d7e11d1c)


---

