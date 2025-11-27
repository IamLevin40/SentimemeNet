# SentimemeNet
CCS 248 Final Project | JZL

## Project Description
SentimemeNet is a deep learning project that implements binary classification models for identifying memes and non-memes using advanced neural network architectures.

## Notebooks

### 1. `meme_vs_not_meme.ipynb` - Meme vs. Non-Meme Binary Classification
A comprehensive notebook implementing a **Mini-ResNet architecture** for binary classification of memes with advanced dataset handling capabilities.

#### Features:
- **Model Architecture**: Mini-ResNet with 6 residual blocks and skip connections
- **Input Size**: 224×224 RGB images
- **Output**: Binary classification (0 = Non-Meme, 1 = Meme)
- **Optimizer**: Adam with learning rate 0.0001
- **Batch Size**: 2
- **Epochs**: 15 with Early Stopping

#### Advanced Dataset Handling:
- **Nested Folder Support**: Automatically scans and collects images from nested subfolders
- **Image Validation**: Validates each image file to ensure TensorFlow compatibility
- **Auto-Filtering**: Automatically filters out corrupted or invalid image files
- **Automatic Balancing**: Ensures equal number of images per class to prevent model bias
- **Smart Sampling**: Randomly selects images from larger class to match smaller class size
- **Configurable Max Limit**: Set `MAX_IMAGES_PER_CLASS` to limit images per class (with smart fallback)
- **Multiple Formats**: Supports .jpg, .jpeg, .png, .bmp, .gif image formats
- **Flexible Structure**: Works with any subfolder organization within meme/ and not_meme/
- **Error Prevention**: Pre-validation prevents training failures from corrupted images

#### Dataset Configuration:
- **Split**: 70% Training / 15% Validation / 15% Testing
- **Augmentation**: Random Rotation (±15°), Horizontal Flip, Random Contrast, Random Zoom, Random Translation
- **Normalization**: Pixel values scaled to [0, 1]
- **Balancing**: Automatic class balancing (min images per class)

#### Training Strategy:
- **Callbacks**: Early Stopping (patience=5), Model Checkpoint
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

#### Model Output:
- Saved model: `models/meme_detector_model.h5`
- Prediction function: `predict_meme(img_path)`
- Automatic cleanup of temporary files

### 2. `resnet_sample.ipynb` - ResNet Sample Implementation
A sample notebook demonstrating ResNet architecture with transfer learning.

## Dataset Structure

### Simple Structure (Flat):
```
meme_vs_not_meme_dataset/
    meme/
        image1.jpg
        image2.png
        ...
    not_meme/
        image1.jpg
        image2.png
        ...
```

### Complex Structure (Nested - Also Supported):
```
meme_vs_not_meme_dataset/
    meme/
        subfolder1/
            image1.jpg
            image2.png
        subfolder2/
            image3.jpg
        image4.jpg        # Mixed structure works too
    not_meme/
        category_a/
            image1.jpg
        category_b/
            subfolder/
                image2.png
        image3.jpg
```

**Note**: The notebook automatically handles both structures and any combination of nested folders!

## Dataset Balancing & Max Limit Configuration

The notebook automatically balances the dataset to prevent model bias and supports configurable maximum images per class.

### Configurable Maximum Images Per Class

Set `MAX_IMAGES_PER_CLASS` in the hyperparameters section:

**Option 1: No Limit (Default)**
```python
MAX_IMAGES_PER_CLASS = None
```
- Uses all available images after balancing

**Option 2: Set a Limit (e.g., 10,000)**
```python
MAX_IMAGES_PER_CLASS = 10000
```
- Limits each class to maximum 10,000 images
- If a folder has fewer images, uses actual count (smart fallback)
- Then balances both classes to ensure equal representation

### Example Scenarios:

**Example 1: Balanced Dataset (No Limit)**
- `meme/`: 5000 images
- `not_meme/`: 5000 images
- `MAX_IMAGES_PER_CLASS = None`
- **Result**: Uses all 5000 images from each class ✓

**Example 2: Imbalanced Dataset (No Limit)**
- `meme/`: 5500 images
- `not_meme/`: 6000 images
- `MAX_IMAGES_PER_CLASS = None`
- **Result**: Randomly selects 5500 images from each class (minimum) ✓

**Example 3: Large Dataset with Limit**
- `meme/`: 15,000 images
- `not_meme/`: 12,000 images
- `MAX_IMAGES_PER_CLASS = 10000`
- **Result**: Limits meme to 10,000, not_meme to 10,000 → Both use 10,000 ✓

**Example 4: Mixed Scenario with Limit**
- `meme/`: 8,000 images
- `not_meme/`: 15,000 images
- `MAX_IMAGES_PER_CLASS = 10000`
- **Result**: Meme uses 8,000 (below limit), not_meme limits to 10,000 → Both use 8,000 (balanced) ✓

### How It Works:
1. Scans all subfolders and counts images in each class
2. Applies `MAX_IMAGES_PER_CLASS` limit to each folder (if configured)
3. If a folder has fewer images than the limit, uses actual count
4. Balances both classes to the minimum count
5. Randomly samples images (seed=42 for reproducibility)
6. Creates a temporary balanced dataset for training
7. Cleans up temporary files after training

This ensures the model doesn't become biased toward the class with more samples!

## Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pathlib
```

### Running the Notebooks
1. Ensure your dataset is in the `meme_vs_not_meme_dataset/` directory
2. Open the desired notebook in Jupyter or VS Code
3. Run all cells sequentially
4. The trained model will be saved to `models/` directory

### Making Predictions
```python
# Load the prediction function from the notebook
predict_meme('path/to/your/image.jpg')
```

## Model Performance
The model is evaluated on:
- **Loss**: Binary Crossentropy
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall

## Project Structure
```
SentimemeNet/
├── meme_vs_not_meme.ipynb          # Main classification notebook
├── resnet_sample.ipynb             # ResNet sample implementation
├── README.md                        # Project documentation
├── meme_vs_not_meme_dataset/       # Dataset directory
│   ├── meme/                       # Meme images
│   └── not_meme/                   # Non-meme images
└── models/                          # Saved models directory
    └── meme_detector_model.h5      # Trained model
```

## Technical Details

### Mini-ResNet Architecture:
1. **Initial Convolution**: 32 filters, 7×7 kernel, stride 2
2. **Max Pooling**: 3×3 pool, stride 2
3. **Residual Block 1-2**: 64 filters each
4. **Residual Block 3-4**: 128 filters each (downsampling)
5. **Residual Block 5-6**: 256 filters each (downsampling)
6. **Global Average Pooling**: Reduces spatial dimensions
7. **Dense Layer 1**: 256 units with ReLU + Dropout (0.5)
8. **Dense Layer 2**: 128 units with ReLU + Dropout (0.3)
9. **Output Layer**: 1 unit with Sigmoid activation

### Key Features:
- **Skip Connections**: Enable better gradient flow and prevent vanishing gradients
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting
- **Global Average Pooling**: Reduces parameters and computational cost

## Authors
JZL - CCS 248 Final Project

## License
This project is for educational purposes as part of CCS 248 coursework.
