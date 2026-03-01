# Cat and Dog Image Classification

**Note:** This repository contains a TensorFlow-based image classifier that can distinguish between cats and dogs using a convolutional neural network trained on the Microsoft COCO and ImageNet datasets.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python train_classifier.py
   ```

3. **Classify an Image**
   ```bash
   python classify_image.py --image_path "path/to/your/image.jpg"
   ```

## Model Details

- **Architecture:** ResNet50 with transfer learning
- **Input Size:** 224x224 pixels
- **Accuracy:** ~95% on test set
- **Datasets Used:** COCO + ImageNet subsets

## Examples

```python
from classifier import CatDogClassifier

clf = CatDogClassifier()
result = clf.predict('cat.jpg')
print(result)  # {'animal': 'cat', 'confidence': 0.98}
```

## Files

- `train_classifier.py` - Training script
- `classify_image.py` - CLI for predictions
- `classifier.py` - Core model class
- `models/` - Trained model artifacts
- `data/` - Sample images for testing

## Results

Model was trained on 5000 cat/dog images with 80-20 train-test split.

## Future Improvements

- Add dog breed classification
- Web UI for predictions
- REST API endpoint
- Mobile app integration

## Notes

- README refreshed and standardized on March 2026.
- Keep this file aligned with actual implementation updates.
