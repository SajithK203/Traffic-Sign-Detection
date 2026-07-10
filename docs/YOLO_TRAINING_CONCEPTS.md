# YOLOv8 Training Concepts & Fundamentals

**CO543/CO5430 — Traffic Sign Detection**
*Prepared as a reference for the Method Section and Final Viva.*

---

## The Big Picture: What is "Fine-Tuning"?

In our Zero-Shot baseline, we saw that the pre-trained `yolov8n.pt` model was terrible at finding traffic signs. That’s because it was trained on the **COCO dataset** (Common Objects in Context), which has 80 classes like "dog", "car", and "person", but only one generic "stop sign" class. It has never seen a German speed limit sign or a warning triangle.

**Fine-Tuning (Transfer Learning)** is the process of taking that "smart" model (which already knows how to see shapes, edges, and objects in general) and teaching it a specific new task (finding 3 types of GTSDB traffic signs) using a relatively small dataset (our 365 training images).

Instead of training a model from scratch for weeks, fine-tuning takes just minutes because the model already has foundational "vision".

---

## The Core Concepts in Notebook 04

When you hit "Run" on the training cell in Notebook 04, here are the fundamentals of what is happening under the hood:

### 1. Epochs and Batches
*   **Epoch:** One complete pass through your entire training dataset (all 365 images). We are running for **50 epochs**, meaning the model will look at the dataset 50 times to learn from it.
*   **Batch Size:** The model doesn't look at one image at a time, nor all 365 at once. It processes them in "batches" (e.g., 16 images at a time). It makes a guess for those 16 images, checks the answers, and adjusts its "brain" before moving to the next 16.

### 2. The Forward Pass (Making Guesses)
For every image, the YOLO (You Only Look Once) neural network looks at the image and outputs thousands of proposed bounding boxes. For each box, it guesses:
1.  **Where is it?** (x, y coordinates, width, height)
2.  **What is it?** (Is it Class 0: Prohibitory, Class 1: Danger, or Class 2: Mandatory?)
3.  **How confident am I?** (0% to 100%)

### 3. The Loss Function (Checking the Answers)
After the model makes its guesses, it checks them against the **ground truth `.txt` label files**. The difference between the model's guess and the correct answer is called the **Loss**. YOLO calculates three types of loss:
*   **Box Loss:** How far off were the coordinates and size of the bounding box? (Are the borders tight around the sign?)
*   **Class Loss:** Did it guess "Danger" when it was actually "Prohibitory"?
*   **Objectness (DFL) Loss:** Did it predict a box where there is no sign at all (a false positive)?

### 4. Backpropagation (Learning from Mistakes)
The goal of training is to make the Loss as close to zero as possible. Once the loss is calculated, the model uses an algorithm called **Backpropagation** (specifically an optimizer like SGD or AdamW). It mathematically goes backward through the neural network and tweaks the internal numbers (called "weights") so that next time it sees that image, it will make a better guess.

### 5. Validation (The "Pop Quiz")
If you only test a student on the exact same homework they practiced on, they might just memorize the answers (this is called **Overfitting**). 
To prevent this, after every single Epoch, YOLO runs a "pop quiz" on the **80 Validation images** (`val` split). The model is *never* allowed to learn from the Validation images. It only predicts on them, and we calculate the **mAP** (mean Average Precision). 

If the model is truly learning general rules about traffic signs (and not just memorizing the 365 training images), the validation mAP will go up over time.

### 6. Saving the "Best" Weights
Because of the validation step, YOLO tracks which epoch scored the highest mAP on the validation set. Even if you train for 50 epochs, maybe Epoch 42 was actually the smartest the model ever got (before it started memorizing too much). 
YOLO automatically saves the weights from that peak moment as `results/checkpoints/yolov8n_gtsdb_best.pt`. **This `.pt` file is our final, trained AI.**

---

## Data Augmentation (The Secret Weapon)

When Notebook 04 runs, YOLO doesn't just show the network the exact same 365 images 50 times. It uses **Data Augmentation** on the fly. 
Every time it loads an image, it slightly alters it:
*   Changes the brightness or contrast (simulating different times of day).
*   Slightly scales or translates the image.
*   Adds a tiny bit of blur or noise.

**Crucially, it does NOT flip the images left-to-right (`fliplr=0` in our config).** If you flipped a "Turn Left" sign, it would look like a "Turn Right" sign, confusing the model!

By slightly altering the images every epoch, those 365 images act like thousands of different images, making the final model much more robust to real-world conditions.

---

## What You Need to Do Now

Since you have `ultralytics` installed, you can go into your Jupyter Notebook tab in your browser, open `notebooks/04_training.ipynb`, and **run all the cells**. 

You will literally see it print out the Epochs 1/50, 2/50, etc., and watch the Box Loss and Class Loss go down, while the mAP (accuracy) goes up!
