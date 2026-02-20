#🍎 Transfer Learning for Identifying Rotten Fruits and Vegetables
An intelligent deep learning project that classifies fruits and vegetables as Fresh or Rotten using Transfer Learning and Convolutional Neural Networks (CNNs). This system helps reduce food wastage and supports quality control in agriculture, supermarkets, and supply chains.

📌 Project Overview
Manual inspection of fruits and vegetables can be time-consuming and inaccurate. This project uses pre-trained deep learning models to automatically detect whether a fruit or vegetable is fresh or rotten from an image.
We leverage Transfer Learning to improve accuracy while reducing training time and computational cost.

🎯 Objectives
Detect whether a fruit/vegetable is Fresh or Rotten
Use pre-trained CNN models (like MobileNetV2 / VGG16)
Build a user-friendly Flask Web Application
Achieve high accuracy with limited dataset
Reduce food wastage using AI automation

🧠 Technology Stack
Python
TensorFlow / Keras
Transfer Learning (MobileNetV2 / VGG16)
Flask
HTML, CSS
NumPy, OpenCV

⚙️ Project Flow
1️⃣ The user uploads an image through the web interface.
2️⃣ The image is preprocessed (resized & normalized).
3️⃣ The trained Transfer Learning model analyzes the image.
4️⃣ The model predicts whether the item is Fresh or Rotten.
5️⃣ The result is displayed on the webpage.

🏗️ Model Architecture
Base Model: MobileNetV2 (Pre-trained on ImageNet)
Frozen convolutional layers
Custom dense layers added
Dropout for regularization
Softmax activation for classification

🚀 How to Run the Project
Install Dependencies
pip install -r requirements.txt

Train the Model (Optional)
python train_model.py

Run the Flask App
python app.py

Open in browser:
http://127.0.0.1:5000/

📊 Model Performance
Accuracy: ~90–95% (depending on dataset)
Loss: Reduced using Adam optimizer
Validation accuracy monitored to avoid overfitting

💡 Advantages
✔ Fast training using Transfer Learning
✔ Works with small datasets
✔ Reduces manual inspection effort
✔ Scalable for real-world deployment

🔮 Future Improvements
Add more fruit & vegetable categories
Deploy on cloud (AWS / Render / Heroku)
Convert to Mobile App
Integrate with IoT camera systems

📌 Applications
Supermarkets
Agriculture quality control
Food supply chain management
Smart farming systems
