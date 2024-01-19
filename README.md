# <h1 align="center"><a href="https://subratamondal1-emotionai-emotionaiapp-7r6pf3.streamlit.app/" target="_blank" rel="noopener noreferrer" >Human Emotion Detector</a></h1>


## Overview
EmotionAI is an interactive web application that utilizes a fine-tuned DistilBERT model to classify emotions from text. It is designed to provide real-time sentiment analysis to users by predicting emotions conveyed in the input text.

## Installation

### Prerequisites
- Python 3.6 or higher
- pip package manager

### Setup
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/subratamondal1/emotionai.git
   ```
2. Navigate to the project directory:
   ```bash
   cd emotionai
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the EmotionAI application, execute the following command in the terminal:

```bash
streamlit run app.py
```

The web application will be hosted locally, and you can interact with it by entering text to analyze the emotions.

## Machine Learning Model
The application uses a DistilBERT model that has been fine-tuned for emotion classification. The model is accessed via the Hugging Face Transformers library and is capable of classifying text into various emotion categories with high accuracy.

## Technologies Used
- Streamlit for web application development
- Hugging Face Transformers for accessing pre-trained models
- Pandas for data manipulation
- Matplotlib for data visualization
- GitHub Actions for CI/CD and automation

## Challenges and Solutions

### Challenge 1: Real-time Inference
Performing real-time inference with a deep learning model can be resource-intensive and slow.

**Solution:**
We optimized the model inference by using a lightweight version of BERT, DistilBERT, which maintains high accuracy while being faster and smaller.

### Challenge 2: User Experience
Creating an intuitive user interface that allows for easy interaction with the machine learning model.

**Solution:**
Streamlit was used to build a user-friendly web interface that enables users to input text and view the emotion classification results in an understandable format.

### Challenge 3: Automation of Workflows
Managing and updating the application and its dependencies can be cumbersome.

**Solution:**
GitHub Actions was employed to automate workflows, including CI/CD pipelines, which facilitated consistent updates and maintenance of the codebase.

## Contributing
We welcome contributions to the EmotionAI project. Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Hugging Face for providing the Transformers library and pre-trained models
- Streamlit for their open-source framework for creating data applications
