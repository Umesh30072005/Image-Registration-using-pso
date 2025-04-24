from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Particle:
    def __init__(self, dimensions):
        # Initialize particle position and velocity with appropriate ranges
        # [tx, ty, rotation_angle, scale]
        self.position = np.array([
            np.random.uniform(-100, 100),  # tx: translation x (-100 to 100 pixels)
            np.random.uniform(-100, 100),  # ty: translation y (-100 to 100 pixels)
            np.random.uniform(-180, 180),  # rotation: -180 to 180 degrees
            np.random.uniform(0.5, 1.5)    # scale: 0.5x to 1.5x
        ])
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class ImageRegistrationPSO:
    def __init__(self, n_particles=50, max_iter=100):  # Increased particles and iterations
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def transform_image(self, image, params):
        # params: [tx, ty, rotation_angle, scale]
        tx, ty, angle, scale = params
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation with border replication
        transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        return transformed
    
    def compute_mse(self, img1, img2):
        # Calculate Mean Squared Error only on overlapping regions
        mask = (img1 != 0) & (img2 != 0)
        if np.sum(mask) == 0:
            return float('inf')
        return np.mean((img1[mask] - img2[mask]) ** 2)
    
    def optimize(self, base_image, moving_image):
        # Convert images to grayscale if they're not already
        if len(base_image.shape) == 3:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        if len(moving_image.shape) == 3:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
        
        # Initialize particles
        dimensions = 4  # tx, ty, rotation, scale
        self.particles = [Particle(dimensions) for _ in range(self.n_particles)]
        
        # PSO main loop
        for iteration in range(self.max_iter):
            for particle in self.particles:
                # Transform moving image using particle's position
                transformed = self.transform_image(moving_image, particle.position)
                
                # Calculate MSE
                mse = self.compute_mse(base_image, transformed)
                
                # Update particle's best position
                if mse < particle.best_score:
                    particle.best_score = mse
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if mse < self.global_best_score:
                    self.global_best_score = mse
                    self.global_best_position = particle.position.copy()
            
            # Update particles with adaptive weights
            w = 0.9 - (0.9 - 0.4) * (iteration / self.max_iter)  # Linearly decreasing inertia
            c1 = 2.0  # Cognitive weight
            c2 = 2.0  # Social weight
            
            for particle in self.particles:
                r1, r2 = np.random.rand(2)
                particle.velocity = (w * particle.velocity +
                                  c1 * r1 * (particle.best_position - particle.position) +
                                  c2 * r2 * (self.global_best_position - particle.position))
                
                # Update position with different ranges for different parameters
                new_position = particle.position + particle.velocity
                new_position[0] = np.clip(new_position[0], -100, 100)  # tx
                new_position[1] = np.clip(new_position[1], -100, 100)  # ty
                new_position[2] = np.clip(new_position[2], -180, 180)  # rotation
                new_position[3] = np.clip(new_position[3], 0.5, 1.5)   # scale
                particle.position = new_position
        
        # Apply best transformation
        best_transformed = self.transform_image(moving_image, self.global_best_position)
        return best_transformed, self.global_best_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_images():
    if 'base_image' not in request.files or 'moving_image' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400
    
    base_file = request.files['base_image']
    moving_file = request.files['moving_image']
    
    if base_file.filename == '' or moving_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    # Save uploaded files
    base_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(base_file.filename))
    moving_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(moving_file.filename))
    
    base_file.save(base_path)
    moving_file.save(moving_path)
    
    # Read images
    base_image = cv2.imread(base_path)
    moving_image = cv2.imread(moving_path)
    
    if base_image is None or moving_image is None:
        return jsonify({'error': 'Could not read images'}), 400
    
    # Perform registration
    pso = ImageRegistrationPSO()
    registered_image, mse = pso.optimize(base_image, moving_image)
    
    # Save registered image
    output_filename = f'registered_{uuid.uuid4().hex}.png'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, registered_image)
    
    # Clean up uploaded files
    os.remove(base_path)
    os.remove(moving_path)
    
    return jsonify({
        'registered_image': output_filename,
        'mse': float(mse)
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
