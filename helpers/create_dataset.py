import numpy as np
import random
import json
from typing import List, Tuple, Dict
import os

class Alphanumeric7x5DatasetGenerator:
    def __init__(self):
        # Base patterns for digits (0-9) and capital letters (A-Z)
        # ASCII values: 0-9 = 48-57, A-Z = 65-90
        self.patterns_7x5 = {
            # Digits 0-9 (ASCII 48-57)
            '0': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            '1': [
                [0,0,1,0,0],
                [0,1,1,0,0],
                [1,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [1,1,1,1,1],
            ],
            '2': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [0,0,0,0,1],
                [0,0,0,1,0],
                [0,0,1,0,0],
                [0,1,0,0,0],
                [1,1,1,1,1],
            ],
            '3': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [0,0,0,0,1],
                [0,0,1,1,0],
                [0,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            '4': [
                [0,0,0,1,0],
                [0,0,1,1,0],
                [0,1,0,1,0],
                [1,0,0,1,0],
                [1,1,1,1,1],
                [0,0,0,1,0],
                [0,0,0,1,0],
            ],
            '5': [
                [1,1,1,1,1],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            '6': [
                [0,0,1,1,0],
                [0,1,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            '7': [
                [1,1,1,1,1],
                [0,0,0,0,1],
                [0,0,0,1,0],
                [0,0,1,0,0],
                [0,1,0,0,0],
                [0,1,0,0,0],
                [0,1,0,0,0],
            ],
            '8': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            '9': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,1],
                [0,0,0,0,1],
                [0,0,0,1,0],
                [0,1,1,0,0],
            ],
            
            # Capital Letters A-Z (ASCII 65-90)
            'A': [
                [0,0,1,0,0],
                [0,1,0,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
            ],
            'B': [
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
            ],
            'C': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            'D': [
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
            ],
            'E': [
                [1,1,1,1,1],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,1],
            ],
            'F': [
                [1,1,1,1,1],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
            ],
            'G': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,0],
                [1,0,1,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            'H': [
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
            ],
            'I': [
                [1,1,1,1,1],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [1,1,1,1,1],
            ],
            'J': [
                [1,1,1,1,1],
                [0,0,0,1,0],
                [0,0,0,1,0],
                [0,0,0,1,0],
                [0,0,0,1,0],
                [1,0,0,1,0],
                [0,1,1,0,0],
            ],
            'K': [
                [1,0,0,0,1],
                [1,0,0,1,0],
                [1,0,1,0,0],
                [1,1,0,0,0],
                [1,0,1,0,0],
                [1,0,0,1,0],
                [1,0,0,0,1],
            ],
            'L': [
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,1],
            ],
            'M': [
                [1,0,0,0,1],
                [1,1,0,1,1],
                [1,0,1,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
            ],
            'N': [
                [1,0,0,0,1],
                [1,1,0,0,1],
                [1,0,1,0,1],
                [1,0,0,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
            ],
            'O': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            'P': [
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
            ],
            'Q': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,1,0,1],
                [1,0,0,1,0],
                [0,1,1,0,1],
            ],
            'R': [
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
                [1,0,1,0,0],
                [1,0,0,1,0],
                [1,0,0,0,1],
            ],
            'S': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,0],
                [0,1,1,1,0],
                [0,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            'T': [
                [1,1,1,1,1],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
            ],
            'U': [
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
            ],
            'V': [
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,0,1,0],
                [0,0,1,0,0],
            ],
            'W': [
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,1,0,1],
                [1,1,0,1,1],
                [1,0,0,0,1],
            ],
            'X': [
                [1,0,0,0,1],
                [0,1,0,1,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,1,0,1,0],
                [1,0,0,0,1],
            ],
            'Y': [
                [1,0,0,0,1],
                [0,1,0,1,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
            ],
            'Z': [
                [1,1,1,1,1],
                [0,0,0,0,1],
                [0,0,0,1,0],
                [0,0,1,0,0],
                [0,1,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,1],
            ],
        }
        
        # Character to ASCII mapping
        self.char_to_ascii = {}
        for char in self.patterns_7x5.keys():
            self.char_to_ascii[char] = ord(char)
    
    def add_noise(self, pattern: List[List[int]], noise_level: float = 0.1) -> List[List[int]]:
        """Add random noise to a pattern"""
        noisy_pattern = [row[:] for row in pattern]  # Deep copy
        
        for i in range(7):
            for j in range(5):
                if random.random() < noise_level:
                    noisy_pattern[i][j] = 1 - noisy_pattern[i][j]  # Flip bit
        
        return noisy_pattern
    
    def add_partial_occlusion(self, pattern: List[List[int]], occlusion_prob: float = 0.05) -> List[List[int]]:
        """Randomly occlude some pixels (set them to 0)"""
        occluded_pattern = [row[:] for row in pattern]
        
        for i in range(7):
            for j in range(5):
                if random.random() < occlusion_prob:
                    occluded_pattern[i][j] = 0
        
        return occluded_pattern
    
    def slight_deformation(self, pattern: List[List[int]]) -> List[List[int]]:
        """Apply slight structural variations while maintaining character integrity"""
        deformed_pattern = [row[:] for row in pattern]
        
        # Randomly choose a type of deformation
        deformation_type = random.choice(['thicken', 'thin', 'shift_pixel'])
        
        if deformation_type == 'thicken':
            # Randomly thicken some lines by adding pixels adjacent to existing ones
            for i in range(1, 6):  # Avoid edges
                for j in range(1, 4):  # Avoid edges
                    if pattern[i][j] == 1 and random.random() < 0.1:
                        # Add pixel to a random adjacent position
                        directions = [(0,1), (0,-1), (1,0), (-1,0)]
                        di, dj = random.choice(directions)
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 7 and 0 <= nj < 5:
                            deformed_pattern[ni][nj] = 1
        
        elif deformation_type == 'thin':
            # Randomly remove some pixels that won't break connectivity
            for i in range(7):
                for j in range(5):
                    if pattern[i][j] == 1 and random.random() < 0.05:
                        # Check if removing this pixel would disconnect the pattern
                        # For simplicity, we'll be conservative and only remove edge pixels
                        adjacent_count = 0
                        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 7 and 0 <= nj < 5 and pattern[ni][nj] == 1:
                                adjacent_count += 1
                        if adjacent_count <= 1:  # Only remove if it has 1 or fewer neighbors
                            deformed_pattern[i][j] = 0
        
        elif deformation_type == 'shift_pixel':
            # Slightly shift some pixels
            for i in range(7):
                for j in range(5):
                    if pattern[i][j] == 1 and random.random() < 0.03:
                        # Try to move this pixel to an adjacent position
                        directions = [(0,1), (0,-1), (1,0), (-1,0)]
                        random.shuffle(directions)
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 7 and 0 <= nj < 5 and pattern[ni][nj] == 0:
                                deformed_pattern[i][j] = 0
                                deformed_pattern[ni][nj] = 1
                                break
        
        return deformed_pattern
    
    def generate_sample(self, char: str, apply_augmentation: bool = True) -> np.ndarray:
        """Generate a single sample for a character"""
        base_pattern = self.patterns_7x5[char]
        
        if not apply_augmentation:
            return np.array(base_pattern, dtype=np.float32)
        
        # Apply random augmentations
        pattern = base_pattern
        
        # Apply deformation (30% chance)
        if random.random() < 0.3:
            pattern = self.slight_deformation(pattern)
        
        # Apply noise (40% chance)
        if random.random() < 0.4:
            noise_level = random.uniform(0.05, 0.15)
            pattern = self.add_noise(pattern, noise_level)
        
        # Apply occlusion (20% chance)
        if random.random() < 0.2:
            occlusion_prob = random.uniform(0.02, 0.08)
            pattern = self.add_partial_occlusion(pattern, occlusion_prob)
        
        return np.array(pattern, dtype=np.float32)
    
    def generate_dataset(self, samples_per_char: int, apply_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete dataset with ASCII labels"""
        characters = list(self.patterns_7x5.keys())
        total_samples = samples_per_char * len(characters)
        
        # Initialize arrays
        X = np.zeros((total_samples, 7, 5), dtype=np.float32)
        y = np.zeros(total_samples, dtype=np.int32)
        
        sample_idx = 0
        
        for char in characters:
            class_index = characters.index(char)
            for _ in range(samples_per_char):
                X[sample_idx] = self.generate_sample(char, apply_augmentation)
                y[sample_idx] = class_index
                sample_idx += 1
        
        # Shuffle the dataset
        indices = np.random.permutation(total_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, filename: str):
        """Save dataset to files"""
        np.savez_compressed(filename, X=X, y=y)
        print(f"Dataset saved to {filename}.npz")
    
    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from file"""
        data = np.load(filename)
        return data['X'], data['y']
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, num_samples: int = 20):
        """Print some samples to console for visualization"""
        print("Sample visualizations:")
        print("=" * 50)
        
        for i in range(min(num_samples, len(X))):
            ascii_val = y[i]
            char = chr(ascii_val)
            print(f"Character: '{char}' (ASCII: {ascii_val})")
            for row in X[i]:
                print(''.join(['█' if pixel > 0.5 else '·' for pixel in row]))
            print("-" * 20)
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[int, int]:
        """Get the distribution of classes in the dataset"""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def ascii_to_char_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get character distribution (more readable than ASCII codes)"""
        char_counts = {}
        unique, counts = np.unique(y, return_counts=True)
        for ascii_val, count in zip(unique, counts):
            char = chr(ascii_val)
            char_counts[char] = count
        return char_counts

# Usage example and dataset generation
def main():
    generator = Alphanumeric7x5DatasetGenerator()
    
    # Generate training dataset (with augmentation)
    print("Generating training dataset...")
    X_train, y_train = generator.generate_dataset(samples_per_char=500, apply_augmentation=True)
    
    # Generate test dataset (with moderate augmentation)
    print("Generating test dataset...")
    X_test, y_test = generator.generate_dataset(samples_per_char=100, apply_augmentation=True)
    
    # Generate validation dataset (clean samples)
    print("Generating validation dataset...")
    X_val, y_val = generator.generate_dataset(samples_per_char=50, apply_augmentation=False)
    
    # Save datasets
    generator.save_dataset(X_train, y_train, "alphanumeric_train_dataset")
    generator.save_dataset(X_test, y_test, "alphanumeric_test_dataset")
    generator.save_dataset(X_val, y_val, "alphanumeric_validation_dataset")
    
    # Print dataset information
    print("\nDataset Information:")
    print("=" * 40)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Input shape: {X_train.shape[1:]} (Height x Width)")
    print(f"Total characters: {len(generator.patterns_7x5)} (0-9, A-Z)")
    print(f"ASCII range: {min(generator.char_to_ascii.values())}-{max(generator.char_to_ascii.values())}")
    
    print("\nCharacter distribution (training set - first 10):")
    char_dist = generator.ascii_to_char_distribution(y_train)
    for i, (char, count) in enumerate(sorted(char_dist.items())):
        if i < 10:  # Show first 10
            print(f"'{char}' (ASCII {ord(char)}): {count} samples")
        elif i == 10:
            print("... and more")
            break
    
    # Show some sample visualizations
    print("\nSample training examples:")
    generator.visualize_samples(X_train, y_train, 8)
    
    # Show clean validation examples
    print("\nSample validation examples (clean):")
    generator.visualize_samples(X_val, y_val, 8)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate datasets
    datasets = main()
    
    print("\nDatasets generated successfully!")
    print("Files created:")
    print("- alphanumeric_train_dataset.npz (training data)")
    print("- alphanumeric_test_dataset.npz (test data)")  
    print("- alphanumeric_validation_dataset.npz (validation data)")
    
    print("\nTo load the datasets in your CNN training code:")
    print("```python")
    print("import numpy as np")
    print("train_data = np.load('alphanumeric_train_dataset.npz')")
    print("X_train, y_train = train_data['X'], train_data['y']")
    print("# y_train contains ASCII values (48-57 for 0-9, 65-90 for A-Z)")
    print("# To convert back to characters: chr(y_train[i])")
    print("```")
    
    print("\nASCII Reference:")
    print("Digits: 0-9 → ASCII 48-57")
    print("Letters: A-Z → ASCII 65-90")