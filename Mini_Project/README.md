# R Programming Mini Project — Image Recognition and Classification with Keras

## Project Title
**Handwritten Digit Recognition with a Dense Neural Network vs a CNN in R (MNIST)**

## Objective
To build a complete deep learning image-recognition pipeline in R using `keras3` — loading image data,
exploring it, pre-processing it, training two different network architectures, comparing them, and
finally using the trained model to recognise digits read back from external `.png` image files.

## Problem Description
The project recognises handwritten digits (0–9) from 28×28 grayscale images. A fully connected
(dense) neural network is trained first as a baseline, then a Convolutional Neural Network (CNN) is
trained on the same data. The two are compared on test loss and accuracy, the better model is
evaluated in detail with a confusion matrix and per-class precision / recall / F1, its mistakes are
inspected visually, and the model is then applied to image *files* on disk — the way a real
recognition system receives its input — before being saved for reuse.

## Reference
Based on ***Image Recognition & Classification with Keras in R*** by Dr. Bharatendra Rai
(Deep Learning with R / Keras playlist) — https://youtu.be/iExh0qj2Ouo

## Dataset
**MNIST handwritten digits**, bundled with Keras (`dataset_mnist()`) — no manual download needed.

| Property | Value |
|---|---|
| Total images | 70,000 grayscale |
| Training set | 60,000 (48,000 train + 12,000 validation via `validation_split = 0.2`) |
| Test set | 10,000 |
| Image size | 28 × 28 × 1 |
| Pixel range | 0 (black background) – 255 (white stroke) |
| Classes | 10 (digits 0–9), near-balanced at 9.0%–11.2% per class |

## R Packages / Libraries Used
| Package | Purpose |
|---|---|
| `keras3` | Building, compiling, training, evaluating and saving the networks; image file I/O |
| `caret` | Confusion matrix and per-class Sensitivity / Specificity / Precision / F1 |
| `ggplot2` | Class distribution, learning curves, model comparison, confusion-matrix heatmap |
| `dplyr`, `tidyr` | Reshaping result tables for plotting |
| `reticulate` | `py_capture_output()` to pull the Keras training log into the notebook |

## Major Operations Performed
1. **Data Loading** — `dataset_mnist()` unpacked with the `%<-%` multi-assignment operator.
2. **Exploratory Analysis** — array shapes and pixel ranges; class-distribution table and grouped bar
   chart; a 5×5 grid of sample digits; the **mean image of each class**; mean-pixel-intensity
   boxplots per digit.
3. **Pre-processing** — `array_reshape()` to 784-vectors (dense) and to `(28, 28, 1)` (CNN);
   scaling to `[0, 1]`; `to_categorical()` one-hot encoding of labels.
4. **Model A — Dense baseline** — 512 → 256 → 10 with ReLU, dropout 0.3 twice, softmax output;
   Adam (lr 1e-3), categorical cross-entropy, 15 epochs, batch size 128.
5. **Model B — CNN** — Conv2D(32, 3×3) → Conv2D(64, 3×3) → MaxPooling(2×2) → Dropout(0.25) →
   Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax); same optimiser, 10 epochs.
6. **Training Analysis** — captured epoch-by-epoch Keras logs, per-epoch history tables and
   `plot(history)` learning curves for both models.
7. **Model Comparison** — side-by-side table and faceted bar chart of parameters, test loss and
   test accuracy.
8. **Detailed Evaluation** — 10×10 confusion matrix, overall Accuracy / Kappa with 95% CI, per-class
   metrics, a confusion-matrix heatmap, the top-10 most confused digit pairs, and a gallery of the
   15 mistakes the model was most confident about.
9. **External Image Recognition** — test digits written to `.png` with `image_array_save()`, then
   read back with `image_load()` / `image_to_array()`, reshaped and classified, scored against the
   true label stored in each file name.
10. **Model Persistence** — `save_model()` to `mnist_cnn.keras`, reloaded with `load_model()` and
    re-evaluated to confirm the reloaded model reproduces the original accuracy.

## Instructions to Execute the Project
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>/Mini_Project
   ```
2. Open `R_Mini_Project_Image_Recognition_Keras_23102A0001.ipynb` in **Google Colab** and set the
   runtime to **R** (Runtime → Change runtime type → R).
3. Run the setup cell. It installs `keras3`, `caret`, `ggplot2`, `dplyr` and `tidyr` if they are
   missing. On a fresh runtime the Python/TensorFlow backend is also needed once — run
   `keras3::install_keras(backend = "tensorflow")` and restart the session.
4. Run all cells. The dataset downloads automatically; no files need to be uploaded.
5. Expect roughly 2–3 s per epoch for Model A and 13–21 s per epoch for Model B on CPU
   (≈ 3 minutes in total); a GPU runtime is faster but not required.

> **Note on the environment:** `keras3` is used rather than the older `keras` package, since `keras3`
> is the current R interface and works with present-day TensorFlow (2.21 here). Keras writes its
> training progress to Python's stdout, which the R kernel does not capture on its own, so the `fit()`
> calls are wrapped in `reticulate::py_capture_output()` to keep the training log inside the notebook.
> `set_random_seed(42)` is set, and re-running reproduced identical metrics.

## Important Results / Output

**Model architectures**
```
Model A — "dense_baseline"                    Model B — "mnist_cnn"
Layer (type)        Output Shape   Param #    Layer (type)         Output Shape      Param #
dense (Dense)       (None, 512)    401,920    conv2d (Conv2D)      (None,26,26,32)       320
dropout (Dropout)   (None, 512)          0    conv2d_1 (Conv2D)    (None,24,24,64)    18,496
dense_1 (Dense)     (None, 256)    131,328    max_pooling2d        (None,12,12,64)         0
dropout_1 (Dropout) (None, 256)          0    dropout_2 (Dropout)  (None,12,12,64)         0
dense_2 (Dense)     (None,  10)      2,570    flatten (Flatten)    (None, 9216)            0
                                              dense_3 (Dense)      (None,  128)    1,179,776
Total params: 535,818 (2.04 MB)               dropout_3 (Dropout)  (None,  128)            0
                                              dense_4 (Dense)      (None,   10)        1,290

                                              Total params: 1,199,882 (4.58 MB)
```

(The layer-name counters continue from Model A into Model B, which is why the CNN's dense layers are
numbered `dense_3` / `dense_4`.)

**Test set evaluation (10,000 images)**
| Model | Parameters | Test loss | Test accuracy | Test error |
|---|---|---|---|---|
| A: Dense 784-512-256-10 | 535,818 | 0.0728 | **98.10%** | 1.90% |
| B: CNN 32/64 conv + dense 128 | 1,199,882 | 0.0305 | **99.21%** | 0.79% |

CNN: Accuracy 0.9921, Kappa 0.9912, 95% CI [0.9902, 0.9937] — **79 misclassified out of 10,000**.

**Per-class metrics (CNN)** — hardest and easiest classes
| Digit | Sensitivity | Precision | F1 |
|---|---|---|---|
| 9 (hardest) | 0.9762 | 0.9970 | 0.9865 |
| 7 | 0.9942 | 0.9836 | 0.9889 |
| 1 (easiest) | 0.9974 | 0.9974 | 0.9974 |

**Most confused digit pairs (CNN)**
| Actual | Predicted as | Count |
|---|---|---|
| 2 | 7 | 8 |
| 9 | 4 | 7 |
| 5 | 3 | 5 |
| 7 | 2 | 5 |
| 9 | 5 | 5 |
| 9 | 7 | 5 |

**Recognition from external `.png` files**
| File | True | Predicted | Correct | Confidence | Runner-up |
|---|---|---|---|---|---|
| digit_004_true0.png | 0 | 0 | TRUE | 1.000000 | 6 (2.41e-08) |
| digit_016_true5.png | 5 | 5 | TRUE | 1.000000 | 3 (3.39e-08) |
| digit_047_true1.png | 1 | 1 | TRUE | 1.000000 | 3 (2.68e-07) |
| digit_102_true0.png | 0 | 0 | TRUE | 1.000000 | 6 (1.32e-09) |
| digit_211_true4.png | 4 | 4 | TRUE | 1.000000 | 9 (6.18e-10) |

All 5 files recognised correctly. Saved model (`mnist_cnn.keras`, 13.8 MB) reloaded and reproduced
test accuracy 0.9921 exactly.

## Conclusion
The CNN reduced the test error rate from 1.90% to 0.79% — a **58% reduction** — while training for
fewer epochs than the dense baseline, because convolution preserves the spatial relationship between
neighbouring pixels that flattening to a 784-vector throws away. The dense model also overfitted
(training accuracy 99.11% against validation 98.03%, with validation loss rising after epoch 8),
whereas the CNN's training and validation curves stayed together (99.06% vs 99.04%). The remaining
errors are concentrated in digit pairs that genuinely share strokes — 2/7, 9/4, 5/3 — and the
model's most confident mistakes are malformed digits a human reader would also hesitate over. A full
discussion, along with limitations and possible extensions, is in section 10 of the notebook.
