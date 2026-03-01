import argparse
import json
from pathlib import Path
from classifier import CatDogClassifier

def main():
    parser = argparse.ArgumentParser(description='Classify cat or dog in image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    parser.add_argument('--output_json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    clf = CatDogClassifier()
    result = clf.predict(args.image_path)
    
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Detected: {result['animal'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities - Cat: {result['probabilities']['cat']:.2%}, Dog: {result['probabilities']['dog']:.2%}")

if __name__ == '__main__':
    main()
