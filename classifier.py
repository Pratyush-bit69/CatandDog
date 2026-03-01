import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
from pathlib import Path

class CatDogClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path or Path('models/cat_dog_classifier.h5')
        self.class_names = ['cat', 'dog']
        self.model = None
        self.loaded = False
        
        if self.model_path.exists():
            self.load_model()
        else:
            self.build_model()
    
    def build_model(self):
        """Build ResNet50-based transfer learning model"""
        base_model = keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(str(self.model_path))
            self.loaded = True
        except:
            self.build_model()
    
    def preprocess_image(self, image_path):
        """Preprocess image for model"""
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image_path):
        """Predict if image is cat or dog"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))
        
        return {
            'animal': self.class_names[class_idx],
            'confidence': confidence,
            'probabilities': {
                'cat': float(predictions[0][0]),
                'dog': float(predictions[0][1])
            }
        }
    
    def train(self, train_data, train_labels, validation_data=None, epochs=20):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.model_path), monitor='val_accuracy', save_best_only=True
            )
        ]
        
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def save(self, path=None):
        """Save model to disk"""
        save_path = path or self.model_path
        Path(save_path).parent.mkdir(exist_ok=True)
        self.model.save(str(save_path))
