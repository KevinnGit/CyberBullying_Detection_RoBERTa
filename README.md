# CyberBullying Using Roberta with feature engineering, text augmentation, and context

A Streamlit-based machine learning application that classifies text for toxicity while detecting the tone/sentiment of messages using RoBERTa embeddings and advanced feature engineering.

## Overview

This application uses a fine-tuned RoBERTa model combined with engineered features to detect toxic content in text. It also analyzes the tone of messages (friendly, mocking, sarcastic, banter) to provide nuanced understanding of user intent.

### Key Capabilities
  - Friendly banter and joking
  - Sarcasm
  - Mocking behavior
  - Profanity detection and censoring
- **OCR Support**: Extract and analyze text from images using Tesseract OCR
- **Cross-platform**: Works on Windows, Linux, and macOS

## Model Performance

- **Accuracy**: 92.0%
- **Macro F1 Score**: 0.914
- **AUC Score**: 0.978
- **Classification Threshold**: 0.60
- **Architecture**: RoBERTa + Feature Engineering + Focal Loss

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- Transformers 4.30.0+
- Streamlit 1.28.0+

For OCR support on Windows:
- Tesseract-OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment
**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Tesseract OCR (Optional, for image text extraction)
**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR\`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Project Structure

```
.
├── app6.py              # Main Streamlit application
├── best_model.pt        # Trained FastClassifier model
├── normalizer.py        # Text normalization utilities
├── config.json          # Model configuration and metrics
├── scaler.pkl           # Scikit-learn StandardScaler for features
├── requirements.txt     # Python dependencies
├── packages.txt         # System package requirements
└── README.md           # This file
```

## Usage

Run the Streamlit application:

```bash
streamlit run app6.py
```

The app will open in your default browser at `http://localhost:8501`

### Features

1. **Text Analysis**: Paste text to analyze for toxicity and tone
2. **Image Analysis**: Upload images to extract and analyze text via OCR
3. **Real-time Prediction**: Get instant predictions with confidence scores
4. **Tone Detection**: Understand the underlying tone of the message

## Model Architecture

### FastClassifier
A multi-layer neural network that combines:
- **Input**: RoBERTa embeddings (768 dims) + engineered features (6 dims)
- **Hidden Layers**: 
  - 768+6 → 512 (BatchNorm, ReLU, Dropout 0.35)
  - 512 → 256 (BatchNorm, ReLU, Dropout 0.25)
  - 256 → 128 (BatchNorm, ReLU, Dropout 0.20)
  - 128 → 64 (ReLU)
- **Output**: 1 (Binary classification with Sigmoid)

### Feature Engineering

The model combines RoBERTa embeddings with engineered features:
- Profanity count and intensity
- Text polarity and subjectivity
- Sentiment indicators
- Sarcasm detection
- Friendly insult patterns

## Configuration

The model configuration is stored in `config.json`:

```json
{
  "best_threshold": 0.60,
  "best_macro_f1": 0.9139,
  "accuracy": 92.0,
  "auc": 0.9782,
  "model": "RoBERTa + Feature Engineering + Focal Loss"
}
```

## Dependencies

See `requirements.txt` for detailed version information:

- **torch**: Deep learning framework
- **transformers**: RoBERTa pre-trained model
- **streamlit**: Web application framework
- **textblob**: Text sentiment analysis
- **better-profanity**: Profanity detection and censoring
- **pytesseract**: OCR text extraction
- **pillow**: Image processing
- **pandas**: Data manipulation
- **scikit-learn**: Feature scaling and utilities

## Troubleshooting

### Tesseract OCR not found
- Ensure Tesseract is installed at the correct path
- On Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Update the path in `app6.py` if installed elsewhere

### CUDA/GPU not detected
- The application automatically falls back to CPU
- GPU acceleration requires compatible NVIDIA GPU and CUDA toolkit

### Model file not found
- Ensure `best_model.pt` and `scaler.pkl` are in the working directory
- These files are required for predictions

## Performance Notes

- First run loads RoBERTa (768MB+) into memory
- Streamlit caches models using `@st.cache_resource` for faster subsequent runs
- GPU recommended for batch processing large texts
- CPU inference takes ~1-2 seconds per text

## Future Improvements

- Support for additional languages
- Fine-tuning on domain-specific datasets
- Batch processing interface
- Visualization dashboards for analytics
- API deployment option

