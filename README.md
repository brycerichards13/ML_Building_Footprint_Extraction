# Automated Building Footprint Extraction from High Resolution Satellite Imagery

## Introduction

This project presents a deep learning based solution for automatically extracting building footprints from high resolution satellite imagery. This was project was created as a group final project for CS 424 Introduction to Deep Learning. Using a U-Net architecture with a pretrained ResNet-50 encoder, the model is trained to segment buildings from **NAIP (National Agriculture Imagery Program)** imagery, with labels derived from **OpenStreetMap (OSM)** polygons. This automated approach provides a scalable and efficient solution for generating crucial geospatial data used in urban planning, disaster response, and environmental monitoring.

**Technology Stack**: Python, PyTorch, Google Colab
* **Libraries**: `segmentation-models-pytorch`, `rasterio`, `geopandas`, `matplotlib`, `ee`, `geemap`, `numpy`

---

## Motivation

Accurate building footprint data is essential for a wide range of applications that are vital for urban development and environmental management. These applications include:

* **Urban Planning and Management**: Understanding population density, planning new infrastructure, and managing city growth.
* **Disaster Response and Management**: Assessing damage after natural disasters, planning emergency response, and identifying safe zones.
* **Environmental Monitoring**: Tracking deforestation, monitoring urban sprawl, and studying the impact of climate change.
* **Infrastructure and Utility Management**: Planning the layout of utilities like water, electricity, and telecommunications.

---

## Dataset

The dataset used for this project consists of high resolution satellite imagery and corresponding building footprint annotations.

* **Imagery**: High resolution (1m/pixel) satellite images over Orange County, CA.
* **Labels**: Rasterized building footprints from OpenStreetMap.
* **Preprocessing**: The raw data was preprocessed into smaller image tiles and corresponding masks suitable for training a deep learning model.
* **Data Source**: Links to the datasets can be found at the bottom of the README file.

### Data Preprocessing Steps

1.  **Data Splitting**: The dataset was divided into training and validation sets to properly evaluate the model's performance.
2.  **Mask Generation**: The OSM label data were converted into binary masks, where pixel values of 1 represent buildings and 0 represent the background.
3.  **Tiling**: The large satellite images and masks were split into smaller tiles (e.g., 512x512 pixels) to be fed into the model.

![A grid of satellite image tiles and their corresponding masks](/screenshots/training_data.png)

---

## Model & Training

* **Architecture**: A U-Net model with a **ResNet-50** encoder (pretrained on ImageNet). The model uses skip connections to preserve high resolution details and a final sigmoid activation function to produce a binary output mask (0 for background, 1 for building).
* **Loss Function**: **Dice Loss** was chosen to effectively handle the potential class imbalance between the number of building pixels and background pixels.
* **Optimizer**: The model was trained using the **Adam** optimizer with an initial learning rate of `1e-4` and a **CosineAnnealingWarmRestarts** learning rate scheduler.
* **Augmentations**: To improve model robustness and prevent overfitting, the following data augmentations were applied during training:
    * Random horizontal and vertical flips
    * Random 90 degree rotations
    * Affine jitter (minor scaling, translation, and rotation)
* **Training Process**: The model was trained for a total of **20 epochs** on 512×512 image patches. The training was completed in two phases: an initial 10 epochs, followed by a resumed session for an additional 10 epochs.

---

## Results

The model's performance was evaluated using the Intersection over Union (IoU) and Dice Coefficient metrics on a validation set.

* **Intersection over Union (IoU)**: **0.74**
* **Dice Coefficient**: **0.66**

An IoU of 0.74 indicates a strong overlap between the predicted footprints and the ground truth labels, demonstrating that the model is highly effective.

![Graphed Loss](/screenshots/loss_vs_epochs.png)

### Visual Results

Here are some examples of the model's predictions on the validation set:

![Loss Vs Epochs 1](/screenshots/input_groundtruth_output_comparison_1.png)
![Loss Vs Epochs 2](/screenshots/input_groundtruth_output_comparison_2.png)
![Loss Vs Epochs 3](/screenshots/input_groundtruth_output_comparison_3.png)


---

## Future Work

* **Larger Dataset**: Training the model on a larger and more diverse dataset could improve its generalization capabilities.
* **Different Architectures**: Experimenting with other state of the art segmentation models (e.g., DeepLabv3+, Mask R-CNN) could lead to further performance improvements.
* **Post-processing**: Implementing post-processing techniques, such as polygon simplification, can help to regularize the predicted footprints and make them more realistic.

## Sources

* **NAIP imagery**: https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery
* **OpenStreetMap building footprints**: https://download.geofabrik.de/
* **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch