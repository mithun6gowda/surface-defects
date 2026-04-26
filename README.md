Project: Steel Surface Defect Detection using Transfer Learning
Overview

Developed an image classification model to detect surface defects in steel using transfer learning. The model classifies defects into six categories: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.

Approach
Used pretrained MobileNetV2 (ImageNet weights)
Applied transfer learning with frozen feature extractor
Trained only the classifier head for domain adaptation
Converted grayscale images → RGB for compatibility

Tech Stack
Python
PyTorch
Torchvision
Matplotlib
Results
Achieved 98.33% validation accuracy

Training loss showed consistent convergence
Model generalised well across defect categories

Key Learnings
Importance of preprocessing in transfer learning
Feature reuse from pretrained models
Handling grayscale → RGB transformation
Freezing strategy to avoid overfitting

Future Improvements
Fine-tuning deeper layers
Using EfficientNet / ResNet
Deploying as a web app (Streamlit)