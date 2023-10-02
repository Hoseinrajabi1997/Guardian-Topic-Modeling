# Guardian-Topic-Modeling
---
# Topic Modeling using BERT, LDA, and Autoencoders

## Description

This project presents a novel approach to topic modeling by merging the strengths of both traditional and state-of-the-art techniques. We combine the structured topic distributions from Latent Dirichlet Allocation (LDA) with the deep contextual embeddings from BERT. The amalgamated data is then compressed using an autoencoder to capture the most salient features. Clustering is applied to this reduced representation, enabling the extraction of coherent and distinct topics.

## Features

- **BERT Embeddings**: Uses the BERT model to generate deep contextual embeddings for the text data.
- **LDA**: Employs Latent Dirichlet Allocation to produce traditional topic distributions.
- **Autoencoder**: Compresses the combined BERT and LDA representations into a latent space, preserving essential information.
- **Clustering**: Applies clustering on the compressed representation to derive meaningful topics.

## Steps to Execute the Model:

Let's break down the steps involved in building a topic modeling framework that combines BERT embeddings, LDA results, and autoencoders:

### 1. Data Preprocessing:
- **1.1** Load the dataset containing the documents you wish to model topics for.
- **1.2** Clean the text data: remove special characters, numbers, and unnecessary whitespace, and convert to lowercase.
- **1.3** Tokenize the cleaned text data, splitting the content into individual words or tokens.

### 2. Topic Modeling with LDA:
- **2.1** Construct a dictionary (a mapping of word IDs to words) and a corpus (the word frequency in documents) using the tokenized data.
- **2.2** Apply the LDA model to the corpus to identify initial topic distributions for each document.
- **2.3** Store the topic distributions for use in the combined representation.

### 3. Embedding with BERT:
- **3.1** Load the pre-trained BERT model for embeddings, preferably a lighter version like `distilbert-base-nli-mean-tokens`.
- **3.2** Pass each document through BERT to obtain deep contextual embeddings.
- **3.3** Store these embeddings for each document.

### 4. Combine LDA and BERT Representations:
- **4.1** For each document, concatenate its LDA topic distribution and BERT embedding to form a combined representation.
- **4.2** Normalize the combined representation to ensure each feature (LDA or BERT) doesn't disproportionately influence subsequent steps.

### 5. Dimensionality Reduction with Autoencoder:
- **5.1** Design an autoencoder architecture suitable for the size of the combined representation. This typically consists of an encoder that reduces dimensionality and a decoder that tries to reconstruct the original input from this reduced representation.
- **5.2** Train the autoencoder on the combined representations until a satisfactory reconstruction error is achieved.
- **5.3** Use the encoder part of the trained autoencoder to transform the combined representations into a compressed latent space.

### 6. Clustering:
- **6.1** Decide on the number of topics (clusters) you want to identify.
- **6.2** Apply a clustering algorithm, such as KMeans, on the latent representations obtained from the autoencoder.
- **6.3** Assign each document to a cluster based on its proximity to cluster centroids.

### 7. Analysis & Interpretation:
- **7.1** For each cluster, inspect the documents or the top terms to understand and label the topic it represents.
- **7.2** Evaluate the quality of topics, using metrics like silhouette score or coherence score, and by manually reviewing the topics.
- **7.3** Fine-tune or adjust parameters as needed and iterate over the process for improved results.

By following these steps, you will have built a topic modeling framework that leverages both traditional (LDA) and modern (BERT) techniques, further refined using autoencoders.

## Files

- `Model.ipynb`: The primary Jupyter notebook containing the implementation and experiments of the topic modeling approach.

## Usage

1. Ensure all dependencies, especially BERT and Gensim, are installed.
2. Load your dataset into the `Model.ipynb` notebook.
3. Follow the steps in the notebook to preprocess the data, generate embeddings, compress with the autoencoder, and perform clustering.
4. Analyze the derived topics from the clustering results.

## Requirements

- gensim==4.1.2
- transformers==4.12.2
- torch==1.10.0
- torchvision==0.11.1
- numpy==1.21.2
- pandas==1.3.3
- scikit-learn==1.0
- sentence-transformers==2.1.0


---
