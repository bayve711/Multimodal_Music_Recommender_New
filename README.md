# Multimodal Music Retrieval System

This project implements a multimodal music retrieval system using the Music4All-Onion dataset. The system combines textual, audio, and visual features to provide comprehensive music search capabilities.

## Features

- Multiple retrieval methods:
  - Text-based (TF-IDF, BERT)
  - Audio-based (MFCC, Spectral Contrast)
  - Visual-based (VGG19, ResNet)
  - Tag-based retrieval
  - Multimodal fusion (Early and Late fusion)
- Interactive web interface for music search
- Comprehensive evaluation metrics
- Support for various similarity measures

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Goldenwert/mmsr_ws24_c.git
cd mmsr_ws24_c
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (will be done automatically on first run):
The application will automatically download required NLTK data (punkt and wordnet) on first run.

## Project Structure

```
mmsr_ws24_c/
├── app.py                    # Streamlit web application
├── backend.py               # Core retrieval system logic
├── filter_tags.py          # Tag processing utilities
├── requirements.txt        # Project dependencies
├── evaluation_results/     # Evaluation metrics and results
├── plots/                  # Generated visualizations
└── data/                  # Dataset files (not included in repo)
    ├── id_information_mmsr.tsv
    ├── id_genres_mmsr.tsv
    ├── id_metadata_mmsr.tsv
    ├── id_tags_dict.tsv
    ├── id_lyrics_tf-idf_mmsr.tsv
    ├── id_lyrics_bert_mmsr.tsv
    ├── id_mfcc_bow_mmsr.tsv
    ├── id_blf_spectralcontrast_mmsr.tsv
    ├── id_vgg19_mmsr.tsv
    └── id_resnet_mmsr.tsv
```

## Usage

1. Start the Streamlit application:
```bash
python -m streamlit run app.py
```

2. Open your web browser and navigate to:
   - Local URL: http://localhost:8501
   - The Network URL will be displayed in the terminal

3. Use the interface to:
   - Search for tracks by artist and song name
   - Select different retrieval methods
   - View similar tracks and their details

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

Major dependencies include:
- numpy>=1.23.0
- pandas>=1.4.0
- scikit-learn>=0.24.2
- torch>=1.9.0
- transformers>=4.11.0
- streamlit>=1.41.0
- nltk>=3.9.0
- librosa>=0.8.1

For a complete list, see `requirements.txt`

## Troubleshooting

1. If you encounter NLTK data errors:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

2. If Streamlit is not found after installation:
```bash
# Make sure you're in your virtual environment, then:
python -m pip install --upgrade streamlit
python -m streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Music4All-Onion dataset providers
- Contributors to the various deep learning models used
- Streamlit and other open-source libraries 
