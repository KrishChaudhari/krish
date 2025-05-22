# Sentiment Analysis using NLP

This project implements a robust sentiment analysis pipeline using Natural Language Processing (NLP) techniques and deep learning. It leverages Keras and TensorFlow for model development, utilizes GloVe pre-trained word embeddings, and includes comprehensive data preprocessing steps. The model is designed for binary sentiment classification (positive/negative) of text data.

## Features

- **Data Preprocessing**: Cleans and processes text data, including stopword removal.
- **Tokenization and Padding**: Converts text to sequences and pads them for model input.
- **Embedding Layer**: Integrates [GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained embeddings for meaningful word representations.
- **CNN-Based Model**: Uses a 1D Convolutional Neural Network for effective feature extraction.
- **Training & Validation**: Splits data, trains for 20 epochs, and visualizes both loss and accuracy.
- **Performance Analysis**: Plots metrics and computes validation loss slope for overfitting/underfitting analysis.
- **Reproducibility**: Random sampling and consistent seeds for experiment stability.

## Project Structure

```
.
├── main.py
├── data/
│   ├── training_cleaned.csv
│   └── glove.6B.100d.txt
├── add_metadata.py
├── history.pkl
└── README.md
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- SciPy

Install dependencies:
```bash
pip install tensorflow numpy matplotlib scipy
```

## Dataset

- **training_cleaned.csv**: Labeled data for sentiment analysis.
  - Column 0: Sentiment label (0 = negative, 4 = positive)
  - Column 5: Text (tweet/message)
- **glove.6B.100d.txt**: Pre-trained GloVe word vectors (download from [here](https://nlp.stanford.edu/projects/glove/)).

## Usage

1. **Prepare Data**
   - Place `training_cleaned.csv` and `glove.6B.100d.txt` in the `./data/` directory.

2. **Run the Script**
   ```bash
   python "main.py"
   ```

3. **Output**
   - Model training progress and metrics are displayed.
   - Training history is saved in `history.pkl`.
   - Plots for loss and accuracy per epoch are shown.

## Key Functions

- `remove_stopwords(sentence)`: Strips out common English stopwords.
- `parse_data_from_file(filename)`: Reads and prepares text-label pairs.
- `train_val_split(sentences, labels, training_split)`: Splits data for training and validation.
- `fit_tokenizer(train_sentences, oov_token)`: Tokenizes text data.
- `seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen)`: Pads/truncates sequences.
- `create_model(vocab_size, embedding_dim, maxlen, embeddings_matrix)`: Builds the CNN-based sentiment classifier.

## Model Architecture

- **Embedding Layer**: Initialized with GloVe vectors (non-trainable)
- **Conv1D Layer**: 128 filters, kernel size 5, ReLU activation
- **Dropout Layer**: 0.5 dropout rate
- **GlobalMaxPooling1D**
- **Dense Layer**: 64 units, ReLU activation
- **Output Layer**: 1 unit, sigmoid activation (binary classification)

## Visualization

- **Loss and Accuracy Curves**: Training and validation metrics plotted per epoch.
- **Validation Loss Slope**: Quantitative indicator of model generalization.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/api/)

## Contact

For questions, suggestions, or contributions, please open an issue or contact [KrishChaudhari](https://github.com/KrishChaudhari).

---

Feel free to modify the README as needed for your project!