# Traffic Sign Recognition

A convolutional neural network (CNN) for recognizing traffic signs from images. This project was developed as part of a master's thesis and includes the complete workflow for training, validating, and using a traffic sign classification model.

## Features

- Traffic sign classification using PyTorch
- Custom CNN architecture with batch normalization and dropout
- Training data augmentation with rotation and color jitter
- Automatic training and validation split
- Best-model checkpoint saving based on validation accuracy
- Top-K predictions with confidence scores
- Automatic use of a CUDA-capable GPU when available

## Recognized traffic signs

The current dataset contains the following 10 classes:

1. Main road
2. Speed limit 80
3. Pedestrian crossing
4. Road works
5. Traffic lights
6. Side road
7. Stop
8. Road narrows
9. No stopping or parking
10. No entry

Class folder names and their display names are defined in `signnames.csv`.

## Project structure

```text
.
|-- dataset/          # Training images organized into class folders
|-- images/           # Example images used for prediction
|-- checkpoints/      # Trained model and generated configuration
|-- model.py          # CNN architecture
|-- train.py          # Model training and validation
|-- predict.py        # Prediction on a single image
|-- signnames.csv     # Mapping between folders and traffic sign names
`-- requirements.txt  # Python dependencies
```

## Requirements

- Python 3.9 or newer
- PyTorch
- torchvision
- Pillow

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd prepoznavanje_prometnih_znakova
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset structure

The dataset must use the folder structure expected by `torchvision.datasets.ImageFolder`. Each subfolder represents one class:

```text
dataset/
|-- glavna_cesta/
|-- ogranicenje_brzine_80/
|-- pjesacki_prijelaz/
|-- radovi_na_cesti/
|-- semafori/
|-- sporedna_cesta/
|-- stop/
|-- suzenje_ceste/
|-- zabrana_zaustavljanja_i_parkiranja/
`-- zabranjen_smjer/
```

The folder names must match the `Folder` column in `signnames.csv`.

## Training

Start training with the default settings:

```bash
python train.py
```

The default configuration uses 30 epochs, a batch size of 32, a learning rate of `0.001`, and 20% of the dataset for validation.

Training parameters can be customized:

```bash
python train.py --data-dir dataset --epochs 50 --batch-size 64 --lr 0.001 --val-fraction 0.2 --seed 42
```

The model with the highest validation accuracy is saved as `checkpoints/best_model.pt`. Training also generates `checkpoints/config.json`, which contains the class labels and model configuration required for prediction.

## Prediction

Run a prediction on an image after training:

```bash
python predict.py --image images/example4.jpg
```

Display a different number of most likely classes:

```bash
python predict.py --image images/example4.jpg --top-k 5
```

Use a custom model checkpoint:

```bash
python predict.py --image images/example4.jpg --checkpoint path/to/best_model.pt
```

Example output:

```text
Image: path/to/image.jpg
Top-3 predictions:
  1. Class 6: Stop (probability 98.75%)
  2. Class 9: No entry (probability 0.84%)
  3. Class 0: Main road (probability 0.21%)
```

## Model architecture

Input images are resized to `32 x 32` pixels. The network consists of three convolutional blocks, each containing convolution, batch normalization, ReLU activation, and max pooling. The extracted features are passed to a fully connected classifier with dropout, which outputs one score for each traffic sign class.

## Author

Developed as part of a master's thesis project on traffic sign recognition using deep learning.
