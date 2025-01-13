# Multimodal Music Retrieval System

This project implements a multimodal music retrieval system using the Music4All-Onion dataset. The system combines textual, audio, and visual features to provide comprehensive music search capabilities.

## Features

- Multiple retrieval methods:
  - Text-based (TF-IDF, BERT)
  - Audio-based (MFCC, Spectral Contrast)
  - Visual-based (VGG19, ResNet)
  - Tag-based retrieval
  - Multimodal fusion (Early and Late fusion)
- Web interface for interactive music search
- Comprehensive evaluation metrics
- Support for various similarity measures

## Installation

1. Create a virtual environment:
```bash
python -m venv mmsr_env

# Activate the environment
# On Windows:
mmsr_env\Scripts\activate
# On macOS/Linux:
source mmsr_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
mmsr_ws24_c/
├── app.py              # Flask web application
├── requirements.txt    # Project dependencies
├── static/            # Static files for web interface
├── templates/         # HTML templates
└── name.tex          # Project documentation
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Select a query song and retrieval methods to find similar tracks

## Retrieval Methods

- **Random Retrieval**: Baseline system
- **TF-IDF Retrieval**: Text-based using TF-IDF vectors
- **BERT Retrieval**: Text-based using BERT embeddings
- **MFCC Retrieval**: Audio-based using MFCCs
- **Spectral Contrast Retrieval**: Audio-based using spectral features
- **VGG19/ResNet Retrieval**: Visual-based using deep features
- **Tag-Based Retrieval**: Using user-generated tags
- **Early Fusion**: Combines TF-IDF and BERT features
- **Late Fusion**: Combines MFCC and VGG19 scores

## Evaluation Metrics

- Accuracy metrics: Precision@k, Recall@k, NDCG@k, MRR
- Beyond-accuracy metrics: Coverage@N, Tag/Genre Diversity@N
- Popularity metrics: Popularity Diversity@N, Average Popularity@N

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=0.24.2
- torch>=1.9.0
- transformers>=4.11.0
- librosa>=0.8.1
- pillow>=8.3.1
- matplotlib>=3.4.3
- jupyter>=1.0.0
- flask>=2.0.1
- scipy>=1.7.1
- sqlalchemy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Music4All-Onion dataset providers
- Contributors to the various deep learning models used
- Flask and other open-source libraries 
