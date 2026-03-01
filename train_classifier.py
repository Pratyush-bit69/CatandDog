import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from classifier import CatDogClassifier
import json

def create_sample_data(num_samples=100):
    """Create synthetic training data for demonstration"""
    X_train = np.random.randn(num_samples, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, num_samples)
    y_train = keras.utils.to_categorical(y_train, 2)
    
    X_test = np.random.randn(num_samples // 5, 224, 224, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, num_samples // 5)
    y_test = keras.utils.to_categorical(y_test, 2)
    
    return (X_train, y_train), (X_test, y_test)

def main():
    print("=== Cat and Dog Classifier Training ===")
    print()
    
    clf = CatDogClassifier()
    print("✓ Model initialized (ResNet50 with transfer learning)")
    print()
    
    print("Creating sample training data...")
    (X_train, y_train), (X_test, y_test) = create_sample_data(num_samples=200)
    print(f"✓ Training set: {X_train.shape[0]} images")
    print(f"✓ Test set: {X_test.shape[0]} images")
    print()
    
    print("Training model...")
    history = clf.train(
        train_data=X_train,
        train_labels=y_train,
        validation_data=(X_test, y_test),
        epochs=5
    )
    print()
    
    clf.save()
    print("✓ Model saved to models/cat_dog_classifier.h5")
    print()
    
    eval_result = clf.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {eval_result[1]:.2%}")
    
    metadata = {
        'model': 'ResNet50',
        'input_shape': [224, 224, 3],
        'classes': ['cat', 'dog'],
        'test_accuracy': float(eval_result[1]),
        'trained_samples': X_train.shape[0]
    }
    
    Path('models').mkdir(exist_ok=True)
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Training complete!")

if __name__ == '__main__':
    main()
