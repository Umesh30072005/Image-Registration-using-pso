document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('registrationForm');
    const progressBar = document.getElementById('progressBar');
    const resultsCard = document.getElementById('resultsCard');
    const originalImage = document.getElementById('originalImage');
    const registeredImage = document.getElementById('registeredImage');
    const mseValue = document.getElementById('mseValue');
    const downloadBtn = document.getElementById('downloadBtn');
    const registerBtn = document.getElementById('registerBtn');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show progress bar
        progressBar.classList.remove('d-none');
        resultsCard.classList.add('d-none');
        registerBtn.disabled = true;
        
        // Create FormData object
        const formData = new FormData(form);
        
        try {
            // Show preview of moving image
            const movingImageFile = document.getElementById('movingImage').files[0];
            if (movingImageFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    originalImage.src = e.target.result;
                };
                reader.readAsDataURL(movingImageFile);
            }
            
            // Send request to server
            const response = await fetch('/register', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Registration failed');
            }
            
            const data = await response.json();
            
            // Display results
            registeredImage.src = `/download/${data.registered_image}`;
            mseValue.textContent = data.mse.toFixed(4);
            
            // Set up download button
            downloadBtn.onclick = function() {
                window.location.href = `/download/${data.registered_image}`;
            };
            
            // Show results card
            resultsCard.classList.remove('d-none');
            
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Hide progress bar
            progressBar.classList.add('d-none');
            registerBtn.disabled = false;
        }
    });
}); 