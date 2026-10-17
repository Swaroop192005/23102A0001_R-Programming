# R Programming Mini Project — Image Recognition and Classification with Keras

## Project Title
**Classifying Aeroplanes vs Cars from JPEG Images using EBImage and Keras in R**

## Objective
To build a complete image classification pipeline in R — reading raw JPEG images, exploring them,
resizing and reshaping them into model-ready features, training a fully connected neural network
with Keras, and evaluating it on held-out images — and then to check **how trustworthy that result
actually is**.

## Reference
Follows ***Image Recognition & Classification with Keras in R | TensorFlow for Machine Intelligence
by Google*** by Dr. Bharatendra Rai — https://youtu.be/iExh0qj2Ouo
(author's R code: https://github.com/bkrai/DeepLearningR)

The notebook is organised around the video's own timestamps:

| Time | Step | Section |
|---|---|---|
| 0:00 | Load Packages | 1 |
| 0:29 | Read Images | 2 |
| 4:08 | Explore | 3 |
| 6:16 | Resize | 4 |
| 7:23 | Reshape | 5 |
| 8:45 | Row Bind | 6 |
| 11:49 | One Hot Encoding | 7 |
| 12:42 | Create Model | 8 |
| 16:02 | Compile | 9 |
| 16:38 | Fit Model | 10 |
| 18:49 | Evaluation and Prediction (train data) | 11 |
| 22:20 | Evaluation and Prediction (test data) | 12 |

Sections 13–14 are an addition of my own (seed stability, leave-one-out cross-validation, training
budget comparison and conclusions).

## Dataset
12 JPEG images, reused from `Lab/EXP_4/images/` in this repository:

| Class | Label | Files |
|---|---|---|
| Aeroplanes | **0** | `p1.jpg` – `p6.jpg` |
| Cars | **1** | `c1.jpg` – `c6.jpg` |

The images arrive at very different resolutions (461×433 up to 5000×3333), which is exactly why the
resize step exists. Split, following the video:

- **Training (10 images):** `p1`–`p5` and `c1`–`c5`
- **Test (2 images):** `p6` and `c6`

Each image is resized to 28×28×3 and flattened to a **2352**-value feature vector.

## R Packages / Libraries Used
| Package | Purpose |
|---|---|
| `EBImage` | `readImage()`, `display()`, `resize()`, `combine()`/`tile()` — reading and preparing images |
| `keras3` | `array_reshape()`, `to_categorical()`, model building, compiling, fitting, evaluating |
| `reticulate` | `py_capture_output()` to pull the Keras training log into the notebook |

`EBImage` is a Bioconductor package, installed with the two lines given in the video description:
```r
install.packages("BiocManager")
BiocManager::install("EBImage")
```

## Major Operations Performed
1. **Load packages** — EBImage + keras3 (TensorFlow backend), reproducible seed.
2. **Read images** — all 12 JPEGs read in a single `for` loop with `readImage()`.
3. **Explore** — `print()` dimensions, `display()` a plane and a car, `summary()` showing pixels
   already scaled to 0–1, RGB intensity `hist()`ograms, and a table of every image's original size.
4. **Resize** — all images to 28×28 (originals kept aside so results can be shown as real pictures).
5. **Reshape** — `array_reshape()` to 28×28×3.
6. **Row bind** — `rbind()` into a 10×2352 training matrix and a 2×2352 test matrix; labels 0/1.
7. **One hot encoding** — `to_categorical()` into two dummy columns.
8. **Create model** — sequential 2352 → 256 (ReLU) → 128 (ReLU) → 2 (softmax).
9. **Compile** — `binary_crossentropy`, `optimizer_rmsprop()`, accuracy.
10. **Fit** — 30 epochs, batch size 32, 20% validation split, with learning curves.
11. **Evaluation & prediction (train)** — loss/accuracy, confusion matrix, per-image probabilities,
    and the misclassified pictures displayed.
12. **Evaluation & prediction (test)** — same on the two held-out images.
13. **Reliability check (my addition)** — same model and split re-run under 10 seeds;
    leave-one-out cross-validation over all 12 images; three training budgets compared.
14. **Conclusion** — what the numbers do and do not show, and how to fix it.

> **Note on API changes:** the video calls `predict_classes()` and `predict_proba()`. Both have been
> removed from Keras, so the notebook uses `predict()` for probabilities and `max.col(...) - 1` for
> the class label — same result. Likewise `keras3` takes `input_shape` on
> `keras_model_sequential()` rather than inside the first `layer_dense()`; the network is identical.
> Keras writes training progress to Python's stdout, which the R kernel does not capture, so `fit()`
> is wrapped in `reticulate::py_capture_output()` to keep the log in the notebook.

## Instructions to Execute the Project
1. Clone this repository and open `Mini_Project/` .
2. Open `R_Mini_Project_Image_Classification_Keras_23102A0001.ipynb` in **Google Colab**, and set
   Runtime → Change runtime type → **R**.
3. Make the images available. The notebook searches `images/`, `../Lab/EXP_4/images/`,
   `Lab/EXP_4/images/` and `/content/images/` — in Colab, upload the 12 files into a folder named
   `images`.
4. Run the setup cell. It installs `BiocManager` + `EBImage` and `keras3` if missing; on a fresh
   runtime also run `keras3::install_keras(backend = "tensorflow")` once.
5. Run all cells. Total runtime is a few minutes on CPU — the networks are small and there are only
   12 images. No GPU needed.

## Important Results / Output

**Model architecture** (matches the parameter count worked out by hand in the video)
```
Layer (type)        Output Shape    Param #
dense (Dense)       (None, 256)     602,368     = 2352 * 256 + 256
dense_1 (Dense)     (None, 128)      32,896     =  256 * 128 + 128
dense_2 (Dense)     (None,   2)         258     =    128 * 2 +   2
Total params: 635,522 (2.42 MB)
```

**Single run (sections 11–12)**
| Data | Images | Loss | Accuracy |
|---|---|---|---|
| Training | 10 | 0.5807 | **0.80** (8 of 10) |
| Test | 2 | — | **0.50** (1 of 2) |

Misclassified in training: `c4.jpg`, `c5.jpg` — both cars read as planes. On test, the car `c6.jpg`
was called a plane with 97.9% confidence. Every error points the same way.

**Reliability check (section 13)**
| Check | Result |
|---|---|
| Training accuracy over 10 seeds | mean 0.94, sd 0.07, range 0.80 – 1.00 |
| Test accuracy over 10 seeds | mean 0.75, sd 0.264 — exactly **0.5 five times and 1.0 five times** |
| **Leave-one-out CV (all 12 images)** | **0.5833 (7 of 12)** vs 0.50 baseline |
| LOOCV predictions | 11 of 12 images predicted "plane", including 5 of the 6 cars |

**Training budget comparison**
| Epochs | Batch | Updates/epoch | LOOCV accuracy | Cars found |
|---|---|---|---|---|
| 30 | 32 | 1 | 0.5833 | 1 of 12 |
| 200 | 32 | 1 | 0.5833 | 3 of 12 |
| 200 | 4 | 3 | **0.6667** | 4 of 12 |

## Conclusion
The pipeline reproduces the video faithfully and the parameter count matches exactly. The accuracy,
however, does not reproduce — and section 13 shows why rather than guessing. With a **2-image test
set**, the only possible scores are 0%, 50% and 100%, and re-running the identical model under 10
seeds returned 50% five times and 100% five times. The video's 100% is one of the lucky draws.

Leave-one-out cross-validation over all 12 images gives **0.5833**, barely above the 0.50 baseline of
always guessing "plane" — which is essentially what the model does, predicting "plane" for 11 of 12
images. Training 20× longer with smaller batches raises this only to **0.6667**, so the 30-epoch
setting is under-trained (batch 32 > 10 images means one gradient update per epoch) but that is not
the real problem: **12 images described by 2352 raw pixel values each** gives 196× more features than
examples, so nothing the network learns transfers to a new picture.

The fix is more images, a convolutional network instead of a dense one, data augmentation, and above
all **transfer learning** from a pre-trained model such as ResNet50 or VGG16 — the standard approach
when only a handful of images are available.
