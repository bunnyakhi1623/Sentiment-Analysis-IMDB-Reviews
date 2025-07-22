# Sentiment-Analysis-IMDB-Reviews
"An end-to-end sentiment classification pipeline for movie reviews, implemented in Google Colab."

## Project Overview
This project demonstrates a machine learning pipeline for classifying movie reviews from the IMDB dataset as either positive or negative. It covers data loading, preprocessing, neural network model building (using TensorFlow/Keras), training with techniques to prevent overfitting and model evaluation. The project is implemented entirely in Google Colab, making it accessible and reproducible without specialized hardware.

## Motivation
- To gain practical experience in Natural Language Processing (NLP) and deep learning.
- To understand the end-to-end workflow of an AI/ML project from data acquisition to model deployment.
- To build a foundational project for my portfolio showcasing text classification skills.

## Dataset
The dataset used is the [IMDB Movie Reviews Dataset] (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
- Contains 50,000 movie reviews.
- Each review is labeled as either 'positive' or 'negative'.
- The dataset is perfectly balanced with 25,000 positive and 25,000 negative reviews.

## Technologies Used
- **Python:** The primary programming language.
- **Google Colab:** Cloud-based Jupyter notebook environment for development and execution.
- **TensorFlow / Keras:** For building and training the deep learning model.
- **Pandas:** For data loading and manipulation.
- **Numpy:** For numerical operations.
- **Scikit-learn:** For data splitting (train_test_split).
- **Matplotlib:** For data visualization (training history plots).

## Project Structure
- `IMDB Dataset.ipynb`: The main Google Colab notebook containing all the code, explanations, and results from data loading to model prediction.
- `IMDB Dataset.csv`: The dataset used in this project (originally inside a ZIP file).

## How to Run This Project
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/YourRepositoryName.git](https://github.com/YourGitHubUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
    (Alternatively, simply download the `Sentiment_Analysis_Project.ipynb` file directly from GitHub.)
2.  **Open in Google Colab:** Go to [Google Colab](https://colab.research.google.com/) and upload the `Sentiment_Analysis_Project.ipynb` notebook (`File` > `Upload notebook`).
3.  **Upload the Dataset:** Once the notebook is open, upload the `IMDB Dataset.csv` file (or its containing ZIP file) to your Colab session. You can use the file browser icon on the left sidebar or the `files.upload()` command within the notebook.
    *If you upload the ZIP file, ensure you run the unzipping code provided in the notebook.*
4.  **Set Runtime Type:** Go to `Runtime` > `Change runtime type` and select `GPU` as the hardware accelerator (recommended for faster training).
5.  **Run All Cells:** Execute all cells in the notebook (`Runtime` > `Run all`) to see the full pipeline in action, from data loading to model predictions.

## Model Performance
The model, a simple neural network with an embedding layer, achieved the following performance on the test set after training with Early Stopping:
- **Test Accuracy:** ~[0.8896]
- **Test Loss:** ~[0.2792]

## Future Improvements
- Experiment with more complex neural network architectures (e.g., LSTMs, GRUs, or Bidirectional RNNs) for potentially higher accuracy.
- Explore transfer learning using pre-trained word embeddings (e.g., Word2Vec, GloVe, FastText) or large language models (BERT, GPT variants).
- Implement a more robust text cleaning pipeline (e.g., handling contractions, stemming/lemmatization).
- Build a simple web application (e.g., using Streamlit or Flask) to deploy the trained model and allow users to input reviews for live sentiment prediction.
- Conduct hyperparameter tuning for better model performance.

## License
This project is licensed under the Apache-2.0 & MIT License.

## Contact
Naga Phanindra Reddy Challa/ Naga Challa
Link to your LinkedIn profile, https://www.linkedin.com/in/c-phanindra-reddy-9aaa99176/
