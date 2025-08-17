# Flow-Based Generative Modeling – Academic Project

This repository contains the implementation of my academic project in machine learning. The project focuses on **flow-based generative models**, specifically **Normalizing Flows** and **Flow Matching**, applied to 2D data distributions.

---

## Access to Source Code

For academic integrity reasons, the full source code of this project is kept in a **private repository**.  
If you are a recruiter, hiring manager, or collaborator and would like to review the code,  
please feel free to [contact me](mailto:ygalnep@gmail.com).  
I will be glad to provide access upon request.

---

## Project Overview

![](https://i.imgur.com/RnyU3L7.png)

Generative models aim to learn transformations from a simple distribution (e.g., Gaussian noise) into a complex data distribution. In this project, I implemented and analyzed two flow-based approaches:

1. **Normalizing Flows**

   * Built from affine coupling layers and permutation layers.
   * Optimized using Maximum Likelihood Estimation and the change-of-variable formula.
   * Capable of modeling complex 2D distributions while keeping invertibility.

2. **Flow Matching**

   * A simpler continuous-time formulation of generative modeling.
   * Approximates the derivative of the flow using neural networks.
   * Supports both unconditional and class-conditional modeling.

The chosen dataset is a 2D point distribution sampled from the **Olympic Rings logo**, allowing clear visualization of the learned flows .

---

## Features

* Implementation of **Affine Coupling Layers** and **Permutation Layers**.
* Full forward and inverse mappings for normalizing flows.
* Computation of log-determinants for likelihood training.
* Training loops for both **Normalizing Flows** and **Flow Matching models**.
* Support for **conditional flows** (e.g., class-specific ring generation).
* Visualization utilities for sampling, trajectories, and loss curves.

---

## Technologies

* **Language**: Python 3
* **Frameworks**: PyTorch, NumPy, Matplotlib
* **Training Tools**: Cosine Annealing LR Scheduler, Adam optimizer

---

## Results

The models successfully learn to map Gaussian noise into the Olympic rings distribution. Key experiments include:

* Loss decomposition into log-probability and Jacobian determinant terms.
* Sampling from trained flows (unconditional and conditional).
* Trajectories of points through the flow transformations.
* Analysis of time quantization and flow reversibility in Flow Matching .
