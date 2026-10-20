# SLEAP Workflow

<div class="hero" markdown>
![SLEAP Workflow](assets/images/workflow.png)
</div>

SLEAP enables you to train deep learning models that automatically track body parts of any animal from video for precise and quantitative analysis of behavioral data. This page walks through the typical end-to-end workflow.

!!! tip "New to SLEAP?"
    Follow along with the hands-on [Tutorial](tutorial/overview.md) to learn each step in detail.

---

## 📁 Phase 1: Setup

### 1. Create a project and import videos

Import video clips from your experiment footage. These will be used to build your training dataset.

- [:octicons-arrow-right-24: Tutorial: Setup](tutorial/setup.md)
- [:octicons-arrow-right-24: Tutorial: Importing new data](tutorial/importing-data.md)
- [:octicons-arrow-right-24: Learning: SLEAP System Overview](learnings/system-overview.md)
- [:octicons-arrow-right-24: Learning: Using the GUI](learnings/gui.md)

### 2. Define the skeleton

List the body parts you want to track and how they connect to each other.

- [:octicons-arrow-right-24: Learning: Skeleton Design](learnings/skeleton-design.md)

---

## 🏷️ Phase 2: Label

### 3. Select frames for labeling

Choose an initial set of frames to label. SLEAP provides sampling methods based on image features to help you pick diverse frames.

- [:octicons-arrow-right-24: Tutorial: Initial Labeling](tutorial/initial-labeling.md)

### 4. Label animal poses

Manually place skeleton body parts on animals in each frame. This is the most time-intensive step, but SLEAP's GUI makes it fast.

- [:octicons-arrow-right-24: Guide: Label Quality Control](guides/label-quality-control.md)
- [:octicons-arrow-right-24: Learning: Prediction-Assisted Labeling](learnings/prediction-assisted-labeling.md)

---

## 🧠 Phase 3: Train

### 5. Train the model

Train a neural network on your labeled frames. SLEAP supports multiple model architectures and training configurations.

- [:octicons-arrow-right-24: Tutorial: Training a Model](tutorial/training-a-model.md)
- [:octicons-arrow-right-24: Guide: Creating a Custom Training Profile](guides/creating-a-custom-training-profile.md)
- [:octicons-arrow-right-24: Guide: Run Training on Colab](guides/run-training-and-inference-on-colab.md)
- [:octicons-arrow-right-24: Guide: Running SLEAP Remotely](guides/running-sleap-remotely.md)
- [:octicons-arrow-right-24: Guide: Instance Size Distribution](guides/instance-size-distribution.md)
- [:octicons-arrow-right-24: Learning: Model Configuration](https://nn.sleap.ai/latest/reference/models/)

### 6. Run inference

Apply the trained model to predict poses on unlabeled frames. Prediction quality depends on label quality, quantity, and training settings.

### 7. Refine and repeat

Inspect predictions, correct errors, and retrain. This human-in-the-loop cycle rapidly improves model accuracy.

- [:octicons-arrow-right-24: Tutorial: Correcting Predictions](tutorial/correcting-predictions.md)

!!! info "Active Learning"
    You typically only need to label **100-500 frames** to get accurate predictions on thousands of frames. Each correction you make improves the model.

---

## 🚀 Phase 4: Deploy

### 8. Process additional videos

Once your model performs well, apply it to all your experiment videos.

- [:octicons-arrow-right-24: Guide: Importing Predictions for Labeling](guides/importing-predictions-for-labeling.md)

### 9. Track identities

Link detections across frames to create continuous tracks for each animal. SLEAP provides several tracking algorithms.

- [:octicons-arrow-right-24: Tutorial: Tracking New Data](tutorial/tracking-new-data.md)
- [:octicons-arrow-right-24: Guide: Tracking and Proofreading](guides/tracking-and-proofreading.md)
- [:octicons-arrow-right-24: Learning: Tracking Mistakes](learnings/main-mistakes-by-tracking.md)

### 10. Proofread tracks

Review tracking results in the GUI and fix any identity swaps or errors.

- [:octicons-arrow-right-24: Tutorial: Proofreading](tutorial/proofreading.md)

### 11. Export for analysis

Export pose data and tracks for downstream analysis in Python, MATLAB, or other tools.

- [:octicons-arrow-right-24: Tutorial: Exporting the Results](tutorial/exporting-the-results.md)
- [:octicons-arrow-right-24: Example Notebooks](notebooks/Analysis_examples.ipynb)

---

## Next Steps

[:octicons-arrow-right-24: Start the Tutorial](tutorial/overview.md) – Step-by-step walkthrough of the complete workflow

[:octicons-arrow-right-24: I'm Done SLEAPing, Now What?](tutorial/i-m-done-sleaping-now-what.md) – What to do after you've finished tracking

[:octicons-arrow-right-24: Migrating to SLEAP 1.5+](guides/migrating-to-sleap-1-5.md) – Upgrade guide for users of older SLEAP versions

[:octicons-arrow-right-24: Skeleton Design](learnings/skeleton-design.md) – Tips for designing effective skeletons

[:octicons-arrow-right-24: Model Configuration](https://nn.sleap.ai/latest/reference/models/) – Choose the right model type for your data
