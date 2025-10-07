---

# 🧠 MRI-to-Synthetic CT Brain Scan Translation using Deep Learning

🚀 **A Capstone Project by Meet Jani**
🎓 *Indian Institute of Technology (IIT) Mandi, in collaboration with Masai School*

---

## 📖 Project Overview

* This project focuses on generating **synthetic CT brain scans from MRI inputs** using **deep learning**.

* MRI offers excellent soft-tissue contrast, whereas CT provides superior visualization of bones and tissue density.

* Obtaining both scans for every patient is costly and exposes patients to radiation.

* To address this, I built multiple **deep learning architectures** — from classical **U-Net** to advanced **Pix2Pix GAN** and **Swin Transformer-based** models — to perform **MRI → CT translation** with high structural and perceptual accuracy.

---

## 🎯 Objectives

✅ Build deep learning models that generate CT-like brain images from MRI scans.

✅ Compare architectures across **MAE, MSE, PSNR, SSIM** metrics.

✅ Design efficient preprocessing and RAM-caching pipelines for faster training.

✅ Deploy an interactive **Gradio web app** for real-time inference and visualization.

---

## 📂 Dataset

**Dataset:** [SynthRAD2023 Brain Task-1](https://synthrad2023.grand-challenge.org)
**Samples:** 180 paired T1-weighted MRI and CT volumes

### 🧩 Preprocessing Pipeline

* 3D MRI/CT volume alignment → 2D paired slice generation
* Intensity normalization per slice
* Optional brain masking for region-specific learning
* Data augmentation (flips, rotations, brightness/contrast, Gaussian noise)
* **In-memory caching** for high-speed data loading during training

---

## 🧠 Model Architectures

| Model               | Generator               | Discriminator | Loss Functions          | Notes                       |
| ------------------- | ----------------------- | ------------- | ----------------------- | --------------------------- |
| **UNet**            | UNet                    | None          | L1                      | Baseline                    |
| **Pix2Pix**         | UNet                    | PatchGAN      | L1 + Adversarial        | Best overall balance        |
| **Pix2Pix-ResUNet** | ResUNet                 | PatchGAN      | L1 + SSIM + Adversarial | Sharper details             |
| **SwinPix2Pix**     | UNet + Swin Transformer | PatchGAN      | L1 + Adversarial        | Captures long-range context |
| **SwinGAN**         | SwinUNet                | PatchGAN      | L1 + SSIM + Adversarial | Transformer-based synthesis |

**Optimizer:** Adam (lr = 0.0002, β₁ = 0.5, β₂ = 0.999)
**Batch Size:** 4–12 slices (GPU-dependent) • **Precision:** Mixed (AMP)

---

## ⚙️ Training & Evaluation

**Losses**

* L1 Loss → pixel-level accuracy
* Adversarial Loss (BCE) → GAN realism
* SSIM Loss → perceptual similarity

**Metrics**

* **MAE**, **MSE** → intensity accuracy
* **PSNR** → image fidelity
* **SSIM** → structural similarity

**Validation Results (Best Models)**

| Model           | PSNR      | SSIM      | MAE       | MSE       |
| --------------- | --------- | --------- | --------- | --------- |
| UNet            | 19.74     | 0.755     | 0.232     | 0.197     |
| **Pix2Pix**     | **23.63** | **0.856** | **0.125** | **0.095** |
| Pix2Pix-ResUNet | 23.89     | 0.851     | 0.117     | 0.088     |
| SwinPix2Pix     | 21.40     | 0.780     | 0.160     | 0.143     |

**Inference Speed**

* GPU ≈ 0.3 s/slice • CPU ≈ 3 s/slice

---

## 🌐 Web App Deployment

**Framework:** Gradio

**Features**

* Upload MRI slice (and optional mask)
* Generate synthetic CT in real-time
* Display MAE, MSE, PSNR, SSIM metrics
* Download results as `.png` or `.nii`

---

## 📊 Results

| Model          | PSNR | SSIM  | Qualitative Observations        |
| -------------- | ---- | ----- | ------------------------------- |
| **Pix2Pix**    | 23.6 | 0.86  | Balanced sharpness & smoothness |
| **ResUNet**    | 23.9 | 0.85  | Sharper edges, slightly slower  |
| **Swin-based** | ~21  | ~0.78 | Noisier due to limited data     |

### 🖼 Visual Comparison

| Input MRI               | Ground Truth CT       | Predicted CT              | Error Map                   |
| ----------------------- | --------------------- | ------------------------- | --------------------------- |
| ![MRI](results/mri.png) | ![CT](results/ct.png) | ![Pred](results/pred.png) | ![Error](results/error.png) |

> 📁 Place these sample images in a `results/` folder within your repo.

---

## 🧩 Innovations & Contributions

✅ **RAM Caching** – accelerated training via in-memory dataset
✅ **Slice-wise Augmentation** – improves robustness to variation
✅ **Comprehensive Metric Pipeline** – MAE, MSE, PSNR, SSIM per-slice & per-patient
✅ **Lightweight Web App** – deployable MRI→CT translator demo

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook capstone-project.ipynb

# 3. Train model (U-Net / Pix2Pix)
#   Run training cells inside the notebook

# 4. Evaluate results
#   Use evaluation cells to compute metrics

# 5. Launch Gradio web demo
#   Run final cell with iface.launch()
```

---

## 🧮 Requirements

* Python ≥ 3.8
* PyTorch with CUDA support
* NumPy • SciPy • scikit-image • scikit-learn
* NiBabel • tqdm • Gradio • pytorch-msssim

---

## 🧭 Future Work

* Extend to 3D volumetric U-Net models
* Integrate multi-modal MRI inputs
* Explore CNN + Transformer hybrids
* Validate clinically via PACS integration

---

## 👨‍💻 Author

**Meet Jani**
📧 [janimeet59@gmail.com](mailto:janimeet59@gmail.com)

🔗 [**LinkedIn**](https://linkedin.com/in/meetjani0) | [**GitHub**](https://github.com/MeetJani0)

🎓 *Minor in Data Science & Machine Learning — IIT Mandi (in collaboration with Masai School)*

---

⭐ If you found this work interesting, feel free to connect or star the repository!

---
