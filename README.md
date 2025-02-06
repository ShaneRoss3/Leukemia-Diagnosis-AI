Leukemia Diagnosis AI
This project applies deep learning to classify leukemia status based on medical data. Built using Python, TensorFlow, and scikit-learn, the model preprocesses the dataset, applies feature scaling, and trains a neural network to achieve 85% accuracy.

Technologies & Tools Used:
Python,
TensorFlow/Keras,
scikit-learn,
Pandas, and
Google Colab.

Project Overview
Data Preprocessing:
Loaded and cleaned the leukemia dataset using Pandas.
Converted categorical labels into numerical values (0 for Negative, 1 for Positive).
Handled potential missing values and scaled numeric features using StandardScaler().

Model Architecture:
A 512-512-1 dense neural network with ReLU activation for hidden layers and sigmoid activation for binary classification.
Implemented L2 regularization and dropout layers (50%) to reduce overfitting.
Used Adam optimizer with binary_crossentropy loss function for optimal convergence.

Training & Optimization:
EarlyStopping to halt training if validation loss stops improving, preventing overfitting.
Split the dataset into 80% training and 20% testing, with further validation on 20% of the training set.

Evaluation:
Achieved 85% accuracy on the test set.
Used a confusion matrix and classification report to analyze performance (precision, recall, and F1-score).
