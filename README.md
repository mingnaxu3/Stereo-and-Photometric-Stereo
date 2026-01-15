# Stereo and Photometric Stereo

This repository contains a software-focused implementation of **stereo vision** and **photometric stereo** pipelines. The project emphasizes clean algorithm design, numerical correctness, and modular Python implementations of classical 3D vision techniques.

## Project Overview

The goal of this project is to reconstruct **3D scene structure** from images using two complementary approaches:

* **Stereo vision**, which estimates depth from image correspondences across viewpoints
* **Photometric stereo**, which estimates surface normals and shape from lighting variation

All algorithms are implemented from scratch using scientific Python libraries, with an emphasis on clarity, correctness, and maintainability.

## Repository Structure

* `stereo.py` — Stereo correspondence and depth estimation
* `photometric_stereo.py` — Surface normal and depth recovery using photometric stereo
* `data/` — Provided image datasets (external download)

## Key Components

### 1. Stereo Vision Pipeline

* Implemented stereo matching to compute **disparity maps** between rectified image pairs.
* Converted disparity values into **depth estimates** using known camera geometry.
* Designed matching logic to balance accuracy and performance using vectorized numerical operations.
* Handled edge cases and invalid disparities to produce stable depth maps.

### 2. Photometric Stereo Pipeline

* Estimated **per-pixel surface normals** from multiple images captured under different lighting conditions.
* Solved linear systems using **least squares** to recover surface orientation.
* Integrated surface normals to reconstruct **height maps** representing object geometry.
* Normalized and validated outputs for numerical stability and visualization.

## Engineering Focus

* Modular, well-documented Python functions for each algorithmic step
* Heavy use of **NumPy vectorization** to improve performance and reduce loop overhead
* Clear separation between data loading, computation, and visualization logic
* Numerical robustness through input validation and normalization

## Technologies Used

* **Python 3**
* **NumPy** for linear algebra and array operations
* **SciPy** for numerical methods
* **Matplotlib** for result visualization
* **Pillow** for image loading and preprocessing

## Setup and Usage

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install numpy scipy matplotlib Pillow
   ```
3. Download and unzip the dataset into the project root directory.
4. Run the provided scripts or notebooks to generate depth maps and surface reconstructions.

## Learning Outcomes

* Implemented end-to-end **3D reconstruction pipelines** using classical vision techniques.
* Strengthened skills in **numerical computing**, **linear algebra**, and **algorithmic optimization**.
* Applied software engineering best practices to research-style codebases.

---

This project demonstrates practical experience building reliable, extensible computer vision systems with a strong emphasis on software engineering quality.
