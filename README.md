# Image Registration using Particle Swarm Optimization (PSO)

A web application that performs image registration using Particle Swarm Optimization (PSO) algorithm. The system can align images that differ in rotation, scale, and translation.

## Features

- Image registration using PSO algorithm
- Handles rotation, scaling, and translation
- Web interface for easy image upload and comparison
- Real-time progress tracking
- MSE (Mean Squared Error) calculation for alignment quality
- Download registered images

## Requirements

- Python 3.7+
- Flask
- OpenCV
- NumPy
- Werkzeug

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-registration-pso.git
cd image-registration-pso
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload images:
   - Base Image: The reference image
   - Moving Image: The image to be aligned

4. Click "Register Images" to start the alignment process

5. View results:
   - Original and registered images
   - MSE value indicating alignment quality
   - Option to download the registered image

## Test Images

The repository includes a script to generate test images:
```bash
python create_test_images.py
```

This creates four test images:
1. Original test image
2. 45-degree rotated version
3. 1.5x scaled version
4. Combined scaled and rotated version

## How It Works

1. The PSO algorithm optimizes four parameters:
   - X translation
   - Y translation
   - Rotation angle
   - Scale factor

2. The fitness function uses Mean Squared Error (MSE) to evaluate alignment quality

3. The algorithm iteratively improves the transformation until convergence

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for image processing capabilities
- Flask for web framework
- NumPy for numerical computations 
