# Real-Time Vehicle Pollution Detection and Tracking using Iterative Deep Learning and YOLO

> A feedback-based iterative deep learning framework that detects polluting vehicles from on-road CCTV/surveillance footage, localizes them with YOLOv5, and extracts their license plates using GenAI — built for real-world deployment on existing traffic infrastructure, not new sensor hardware.

Published as: *"Real Time Vehicle Pollution Detection and Tracking using Iterative Deep Learning and YOLO"* — M. Prasha Meena, S.A. Aashish, P. Praveen, T.K. Tilesh, Dept. of Information Technology, Mepco Schlenk Engineering College.

---

## Why this exists

Most vehicle-pollution monitoring today is sensor-based — physical emission sensors bolted onto vehicles or fixed at checkpoints. That works, but it doesn't scale: it's expensive to install, expensive to maintain, and impossible to retrofit onto the millions of vehicles already on the road.

Meanwhile, most cities already have CCTV/traffic-surveillance cameras watching every major road. This project asks a simpler question: **can we detect a polluting vehicle just by looking at it, the same way a traffic officer visually spots black smoke?**

The answer, per the paper, is yes — with an accuracy that beats seven established pretrained CNN baselines once a purpose-built model and an iterative training loop are added on top.

## What the system actually does

Given raw CCTV video of traffic, the pipeline:

1. **Extracts frames** from the video stream.
2. **Detects vehicles** in each frame using **YOLOv5**.
3. **Crops each detected vehicle** out of the frame using **OpenCV**.
4. **Classifies each cropped vehicle** as *polluting* / *not polluting* using an ensemble of 8 models (7 pretrained CNNs + 1 custom model), combined via majority voting.
5. If a vehicle is flagged as polluting, **extracts its license plate** using **Gemini AI**.
6. **Logs the plate number + timestamp to Firebase**, for downstream enforcement.

```
Surveillance → Frame Extraction → Vehicle Detection (YOLOv5) → Pollution Classification
                                                                        │
                                                          ┌─────────── Yes ───────────┐
                                                          │                           │
                                                     Track Number Plate         Ignore (No)
                                                       (Gemini AI)
                                                          │
                                                     Store in Firebase
```

## The three contributions on top of the baseline

The project builds on an existing 7-model pollution-classification framework (Inception-V3, MobileNet-V2, MobileNet-V3, InceptionResNet-V2, VGG16, VGG19, XceptionNet) and adds:

| # | Contribution | Why it matters |
|---|---|---|
| 1 | **A custom CNN model**, purpose-built for this task (10 conv layers, dropout, dense layers, a concatenate layer, average pooling, flatten) | Outperforms all 7 general-purpose pretrained models on both accuracy and F1 — because it's trained on and shaped for *this specific problem*, not repurposed from ImageNet classification |
| 2 | **Video-level processing** — YOLOv5 + OpenCV integrated into the pipeline | Moves the framework from single-image classification to real-time video streams, which is what surveillance cameras actually produce |
| 3 | **License-plate tracking via Gemini AI → Firebase** | Closes the loop from *detection* to *enforcement* — a flagged vehicle isn't just logged, it's identifiable |

## The core idea: Iterative Deep Learning

Labeled pollution data is scarce — nobody has a large, clean, pre-labeled dataset of "smoking vs. non-smoking vehicles." So instead of the usual train-once-and-deploy approach, the framework trains in a loop:

1. Start with a **small labeled set** and a **much larger unlabeled set**.
2. Train all 8 models on the labeled set.
3. Run all 8 models on the unlabeled images. If **≥5 of 8 models agree** (majority vote) **and** every one of them is above a **confidence threshold θ\* = 0.7**, that image gets pseudo-labeled and pulled into the training set.
4. Retrain on the now-larger training set.
5. Repeat until a max iteration count or a max training-set-size cap (ϵ) is hit.

This is effectively **self-training / pseudo-labeling with an ensemble confidence gate**, which is exactly what you want when labeled data is the bottleneck rather than raw data volume. DATASET1 was iterated 4 times, DATASET2 3 times.

## Dataset

- **3,000 vehicle images**, hand-curated to vary in lighting, weather, and camera angle — sourced from Google Images and real captures from New Town Road, Kolkata.
- **Augmentations**: blur (max 15×15 kernel), rotation (±15°), horizontal flip, synthetic rain/fog (via the `imgaug` library), and **night-mode generation via a trained Style-GAN** (500 synthetic night images, since real nighttime pollution samples are scarce).

| | Day – Pollution | Day – Non-pollution | Night – Pollution | Night – Non-pollution |
|---|---|---|---|---|
| **Count** | 1500 | 1000 | 275 | 225 |

Public dataset reference used for comparison in the paper: [VehicleSmokeDataset (GitHub)](https://github.com/srimantacse/VehicleSmokeDataset)

## Results

- **Proposed custom model: >90% accuracy** after 5 epochs, consistently the best of all 8 models on both accuracy and F1 score.
- Existing pretrained baselines (VGG16/19, Xception, ResNet50, MobileNetV3-Small, InceptionV3, InceptionResNetV2): **75–85% accuracy**.
- **Zero-shot evaluation on synthetic rain/fog images: 79.5% accuracy** — beating comparison methods in the same degraded-visibility setting.
- **Inference speed: ~0.3 seconds/image** for the ensemble on PyCharm — fast enough to be paired with real-time camera feeds, and parallelizable further.
- Interpretability was checked with **GradCAM and SmoothGrad**, confirming the model attends to the vehicle and its exhaust region rather than background artifacts.

## Repository contents

| File | What it is |
|---|---|
| `IEEE_Paper_-_YOLO.pdf` | The published paper — full methodology, architecture diagrams, and results tables |
| `CustomModel_Creation___Testing.ipynb` | Defines and trains the custom CNN (`CustomModel`), plus test-accuracy evaluation |
| `Prediction_of_Vehicle_and_Extraction_of_License_Plate.ipynb` | Loads the trained model ensemble, runs majority-vote prediction on a sample image, and calls Gemini to extract the license plate |
| `Vehicle_Smoke_Code.docx` | Supplementary implementation notes — frame extraction (OpenCV), YOLO-based object detection, and related pipeline code |
| `Copy_of_Smoke_Output.avi` | Sample output video showing detected vehicles with "Polluted" / "Not Polluted" bounding-box labels |

> **Note:** the notebooks and scripts here are exactly as used for the paper's experiments (Google Colab, GPU runtime, paths under `/content/drive/...`). They are shared as-is for reproducibility and reference — no refactoring or path cleanup applied.

## Model architecture (custom model)

- **Input:** 299×299×3 RGB image
- **10 convolutional layers** — first 3 with 18/19/20 filters, remaining 7 with 64 filters each
- **2 max-pooling layers** for spatial downsampling
- **1 concatenate layer** merging feature maps from the last two conv layers
- **1 average-pooling layer** for further dimensionality reduction
- **Flatten → Dense(1024) → Dense(1, sigmoid)** for binary polluting/non-polluting classification

## Tech stack

`Python` · `TensorFlow / Keras` · `YOLOv5` · `OpenCV` · `Google Gemini AI` · `Firebase` · `Style-GAN` (for synthetic night-mode augmentation) · Google Colab (GPU)

## Limitations (from the paper's own discussion)

- **Invisible smoke and dense fog** remain hard cases — visible-smoke-based detection has an inherent ceiling.
- Night-mode robustness depends on the diversity of the Style-GAN styler references used during synthetic generation.
- **Motion blur and frame occlusion** can degrade license-plate extraction and vehicle detection quality.
- License-plate tracking in this codebase is a **proof of concept**, not a production enforcement pipeline — it was explicitly noted as being "beyond the current research scope" for full deployment.
- Confidence threshold θ\* (0.7) is currently fixed manually; the paper flags **automatic threshold tuning** as future work.

## Roadmap (from the paper's future work)

- Expand the labeled dataset with more diverse real-world captures.
- Improve tracking with better DL-based surveillance methods.
- Explore color-based pollution classification with lightweight/shallow networks.
- Move the iterative training loop to distributed compute (Spark/Ray) to cut iteration time.
- Automate confidence threshold (θ\*) selection instead of hand-tuning it.

## Addressing likely critiques

A few objections come up naturally when reviewing this kind of system. Here's how the design holds up against each:

**"Pseudo-labeling can reinforce its own mistakes — isn't self-training risky?"**
Yes, in general. That's exactly why the loop isn't a plain self-training scheme: an unlabeled image only enters the training set if **at least 5 of 8 independently-trained models agree** *and* every one of them clears the **0.7 confidence threshold**. A single overconfident model can't corrupt the training set on its own — the majority-vote gate is specifically there to suppress that failure mode. The ablation study in the paper backs this up: the iterative approach outperforms plain one-shot training across all 8 methods, not just the custom model.

**"3,000 images is a small dataset for a deep learning claim."**
It's small in absolute terms, but the framework is explicitly designed for the low-labeled-data regime (that's the whole point of the iterative loop — it treats scarce labels as the default assumption, not an edge case). The heavy augmentation pipeline (blur, rotation, flip, synthetic rain/fog, GAN-generated night images) and the pseudo-labeling loop are both direct responses to that constraint, not workarounds bolted on after the fact.

**"Why build a custom CNN instead of just fine-tuning a bigger pretrained model?"**
Because the comparison in the paper is apples-to-apples: the same 8-model ensemble methodology was applied uniformly, and the custom model — despite being smaller and purpose-built — beat every pretrained general-purpose model (Inception-V3, VGG16/19, Xception, ResNet50, MobileNet-V2/V3, InceptionResNetV2) on both accuracy and F1. The result argues that a small model tailored to the specific texture/color signature of vehicle smoke outperforms features learned for generic ImageNet classification, at a fraction of the parameter count.

**"How do you know it isn't just learning background/road cues instead of actual smoke?"**
This was checked directly with **GradCAM and SmoothGrad** — both confirm the model's attention concentrates on the vehicle and its exhaust region, not on unrelated background pixels.

**"License plate tracking → Firebase sounds like a privacy/surveillance concern."**
Fair, and worth being upfront about: the plate-extraction step only fires for vehicles already classified as polluting, and the paper itself scopes plate-tracking as a proof-of-concept for potential enforcement workflows, not a deployed production system. Any real-world deployment would need to address data retention, access control, and legal authorization for enforcement action separately from the detection model itself.

## Citation

If you use this work, please cite:

```
M. Prasha Meena, S.A. Aashish, P. Praveen, T.K. Tilesh,
"Real Time Vehicle Pollution Detection and Tracking using Iterative Deep Learning and YOLO,"
Department of Information Technology, Mepco Schlenk Engineering College, Sivakasi, Tamil Nadu.
```

## Authors

- Mrs. M. Prasha Meena — Department of Information Technology, Mepco Schlenk Engineering College
- S.A. Aashish
- P. Praveen
- T.K. Tilesh
