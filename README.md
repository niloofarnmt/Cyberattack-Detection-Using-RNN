
This project aims to detect abnormal behaviors in network traffic and identify cyberattacks such as DDoS, port scanning, and other threats using Recurrent Neural Networks (RNN). This document outlines the steps for data collection, preprocessing, model design, training, and evaluation.

2. Project Phases
A. Data Collection
Dataset: The project uses the CICIDS 2017 dataset, which contains labeled network traffic data for various attack types.
Key Features:
Time: Timestamp of the traffic.
Source & Destination IP: Origin and target of the traffic.
Protocol: The communication protocol used.
Packet Count: Number of packets transferred.
Attack/Benign Labels: Classification of traffic as malicious or normal.
B. Data Preprocessing
Removing incomplete and noisy data.
Selecting relevant features and transforming data into time sequences (e.g., 100 traffic samples as a single input for the RNN).
Normalizing data using techniques such as StandardScaler.
Splitting data into training (80%) and testing (20%) sets.
C. RNN Model Design
LSTM Layers: To capture temporal relationships in the data.
Proposed Architecture:
Input: Processed network traffic features.
Layers:
Two LSTM layers with 64 and 32 neurons.
Dropout layers to prevent overfitting.
Dense layer with one output neuron for binary classification (attack or benign).
Built using TensorFlow or PyTorch libraries.
D. Model Training
Optimization: Using the Adam optimizer and appropriate weight initialization.
Hyperparameter Tuning:
Number of epochs.
Batch size.
Learning rate.
Evaluate the model on test data using various metrics.
E. Model Evaluation
Confusion Matrix: To visualize the performance of attack classification.
Evaluation Metrics:
Accuracy: Percentage of correct predictions.
Precision: Accuracy of attack detection.
Recall: Proportion of actual attacks correctly identified.
F1-Score: Harmonic mean of Precision and Recall.
ROC-AUC: Model's ability to distinguish between attacks and benign traffic.
F. Result Analysis
Reviewing misclassified samples where the model made errors.
Analyzing which features contributed to correct or incorrect predictions.
Identifying specific patterns in data where the model failed.
3. Results and Achievements
Model Accuracy: The results demonstrated that the RNN architecture effectively detected cyberattacks with high accuracy.
Errors: The model struggled with certain ambiguous attack patterns or benign traffic resembling attacks.
Key Features: Features such as source/destination IP and packet count were crucial for attack identification.
4. Suggested Improvements
Combining RNN with CNN for better feature extraction.
Using more diverse datasets or real-world traffic data for training.
Applying regularization techniques and fine-tuning hyperparameters.
Leveraging Data Augmentation to create more samples of rare attacks.
5. Tools and Technologies
Libraries: TensorFlow, PyTorch, Scikit-learn, Pandas, Matplotlib.
Programming Language: Python.
Data Source: CICIDS 2017 Dataset.
6. Project Code Documentation
The project's code and related files are available in the following GitHub repository: GitHub Link - CICIDS RNN Project
