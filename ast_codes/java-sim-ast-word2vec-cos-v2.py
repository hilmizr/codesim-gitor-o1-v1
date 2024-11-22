import os
import io
from gensim.models import Word2Vec  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import time 
from tqdm import tqdm  
from datetime import datetime  
import json  
import logging  # Enhanced logging
import seaborn as sns  # For confusion matrix visualization
import matplotlib.pyplot as plt  # For plotting
from sklearn.decomposition import PCA  # For embedding visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
        # You can add FileHandler here to log to a file if desired
    ]
)

# Function to preprocess and tokenize code
def preprocess_code(code_string):
    """
    Tokenizes the code by splitting on whitespace.
    
    Args:
        code_string (str): The raw code snippet.
    
    Returns:
        list: A list of tokens.
    """
    return code_string.split()

# Function to get Word2Vec embedding
def get_word2vec_embedding(model, code_tokens):
    """
    Generates a Word2Vec embedding by averaging word vectors and normalizing the result.
    
    Args:
        model (Word2Vec): Trained Word2Vec model.
        code_tokens (list): Tokenized code.
    
    Returns:
        numpy.ndarray: The normalized embedding vector.
    """
    vectors = [model.wv[word] for word in code_tokens if word in model.wv]
    if vectors:
        embedding = np.mean(vectors, axis=0)
        # Normalize the embedding to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    else:
        return np.zeros(model.vector_size)

# Function to calculate similarity (unchanged)
def calculate_similarity(embedding1, embedding2):
    """
    Calculates the cosine similarity between two embeddings.
    
    Args:
        embedding1 (numpy.ndarray): The first embedding vector.
        embedding2 (numpy.ndarray): The second embedding vector.
    
    Returns:
        float: The cosine similarity score between -1 and 1.
    """
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    return similarity

# Functions for saving and loading embeddings (modified for JSON)
def save_embedding(embedding, save_path):
    """
    Saves the embedding to the specified path in JSON format.
    
    Args:
        embedding (numpy.ndarray): The embedding vector to save.
        save_path (str): The file path where the embedding will be saved.
    """
    try:
        # Convert NumPy array to list for JSON serialization
        embedding_list = embedding.tolist()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_list, f)
    except Exception as e:
        logging.error(f"Error saving embedding to '{save_path}': {e}")

def load_embedding(load_path):
    """
    Loads the embedding from the specified JSON path.
    
    Args:
        load_path (str): The file path from where the embedding will be loaded.
    
    Returns:
        numpy.ndarray: The loaded embedding vector.
    """
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            embedding_list = json.load(f)
        # Convert list back to NumPy array
        embedding = np.array(embedding_list)
        return embedding
    except Exception as e:
        logging.error(f"Error loading embedding from '{load_path}': {e}")
        return None

def ensure_directory(path):
    """
    Ensures that the directory exists; if not, creates it.
    
    Args:
        path (str): The directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)

# Function to get or create embedding using Word2Vec
def get_embedding(file_path, embedding_base_path, relative_path, word2vec_model):
    """
    Retrieves the embedding for a given file. If the embedding does not exist, it generates and saves it.
    
    Args:
        file_path (str): The path to the code file.
        embedding_base_path (str): The base directory where embeddings are stored.
        relative_path (str): The relative path within the embeddings directory corresponding to the code file.
        word2vec_model (Word2Vec): Trained Word2Vec model.
    
    Returns:
        numpy.ndarray: The embedding vector.
    """
    # Modify the embedding filename to exclude AST and Word2Vec as they are in the path
    # Example: original_file.json
    embedding_filename = f"{relative_path}.json"
    
    # Construct the full embedding path
    embedding_path = os.path.join(embedding_base_path, embedding_filename)
    
    # Ensure the directory exists
    embedding_dir = os.path.dirname(embedding_path)
    ensure_directory(embedding_dir)

    if os.path.exists(embedding_path):
        # Load existing embedding
        embedding = load_embedding(embedding_path)
        if embedding is None:
            logging.warning(f"Failed to load embedding from '{embedding_path}'.")
    else:
        # Read the code file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logging.error(f"Error reading file '{file_path}': {e}")
            return None
        # Preprocess and tokenize the code
        tokens = preprocess_code(code)
        # Generate embedding
        embedding = get_word2vec_embedding(word2vec_model, tokens)
        # Save embedding
        save_embedding(embedding, embedding_path)
    return embedding

def visualize_embeddings(embeddings, labels):
    """
    Visualizes embeddings using PCA.
    
    Args:
        embeddings (list of numpy.ndarray): List of embedding vectors.
        labels (list of str): Corresponding labels ('plagiarized', 'non-plagiarized', 'original').
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    unique_labels = list(set(labels))
    colors = ['red', 'green', 'blue']
    for label, color in zip(unique_labels, colors):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label, alpha=0.6, color=color)
    plt.legend()
    plt.title("PCA of Word2Vec Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(TP, FP, FN, TN, threshold, save=False, save_path='confusion_matrix.png'):
    """
    Plots the confusion matrix using seaborn's heatmap.
    
    Args:
        TP (int): True Positives
        FP (int): False Positives
        FN (int): False Negatives
        TN (int): True Negatives
        threshold (float): The similarity threshold used
        save (bool): Whether to save the plot as an image file
        save_path (str): The file path to save the image
    """
    cm = np.array([[TP, FP],
                   [FN, TN]])
    
    labels = ['Plagiarized', 'Non-Plagiarized']
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix at Threshold {threshold}')
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix saved to '{save_path}'.")
    plt.show()

def main():
    # Start total execution timer
    total_start_time = time.time()

    # Generate current datetime string in YYYYMMDD_HHMMSS format
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the path to the IR-Plag-Dataset folder
    dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
        return

    # Define the base path for embeddings with hierarchical directories
    # embeddings/AST/Word2Vec/{datetime}/
    embedding_base_path = os.path.join(os.getcwd(), "embeddings", "AST", "Word2Vec", current_datetime)
    ensure_directory(embedding_base_path)

    # Collect all Java files for training Word2Vec
    all_code = []
    java_file_paths = []  # To store paths for progress bar
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                java_file_paths.append(file_path)

    if not java_file_paths:
        logging.error("No Java files found in the dataset to train Word2Vec model.")
        return

    # Preprocess all Java files with progress bar
    logging.info("Preprocessing Java files for Word2Vec training...")
    start_time = time.time()
    for file_path in tqdm(java_file_paths, desc="Preprocessing Files", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tokens = preprocess_code(code)
            all_code.append(tokens)
        except Exception as e:
            logging.error(f"Error reading file '{file_path}': {e}")
    end_time = time.time()
    logging.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds.\n")

    # Train Word2Vec model
    logging.info("Training Word2Vec model...")
    start_time = time.time()
    word2vec_model = Word2Vec(sentences=all_code, vector_size=100, window=5, min_count=1, workers=4)
    end_time = time.time()
    logging.info(f"Word2Vec model training completed in {end_time - start_time:.2f} seconds.\n")

    # Define a list of similarity thresholds to iterate over
    similarity_thresholds = [0.1, 0.5, 0.52]

    # Initialize variables to keep track of the best result
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f_measure = 0
    best_TP = 0
    best_FP = 0
    best_FN = 0
    best_TN = 0

    # Get list of folders to iterate with progress bar
    folder_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    # To collect embeddings and labels for visualization
    all_embeddings = []
    all_labels = []

    # Loop through each similarity threshold and calculate accuracy
    for SIMILARITY_THRESHOLD in tqdm(similarity_thresholds, desc="Processing Thresholds", unit="threshold"):
        logging.info(f"\nProcessing for Similarity Threshold: {SIMILARITY_THRESHOLD}")
        # Start timer for each threshold
        threshold_start_time = time.time()

        # Initialize the counters for the current threshold
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives
        total_cases = 0

        # Loop through each folder with progress bar
        for folder_name in tqdm(folder_names, desc="Processing Folders", unit="folder", leave=False):
            folder_path = os.path.join(dataset_path, folder_name)
            # Path to the original Java file
            original_path = os.path.join(folder_path, 'original')
            if not os.path.exists(original_path):
                logging.warning(f"Original folder missing in '{folder_path}'. Skipping.")
                continue

            java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            if len(java_files) != 1:
                logging.error(f"Found {len(java_files)} Java files in '{original_path}' for '{folder_name}'. Expected exactly 1. Skipping.")
                continue

            java_file = java_files[0]
            original_file_path = os.path.join(original_path, java_file)

            # Define relative path for embedding
            original_relative_path = os.path.join(folder_name, 'original', os.path.splitext(java_file)[0])

            # Get or create embedding for the original code
            embedding1 = get_embedding(original_file_path, embedding_base_path, original_relative_path, word2vec_model)
            if embedding1 is None:
                logging.warning(f"Failed to obtain embedding for '{original_file_path}'. Skipping comparisons.")
                continue

            # Collect embeddings and labels for visualization
            all_embeddings.append(embedding1)
            all_labels.append('original')

            # Loop through 'plagiarized' and 'non-plagiarized' subfolders
            for subfolder_name in ['plagiarized', 'non-plagiarized']:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    logging.warning(f"Subfolder '{subfolder_name}' missing in '{folder_path}'. Skipping.")
                    continue

                # Collect all Java files in the subfolder
                comparison_files = []
                for root_dir, dirs, files in os.walk(subfolder_path):
                    for java_file in files:
                        if java_file.endswith('.java'):
                            comparison_file_path = os.path.join(root_dir, java_file)
                            comparison_files.append(comparison_file_path)

                if not comparison_files:
                    logging.warning(f"No Java files found in '{subfolder_path}'. Skipping.")
                    continue

                # Process comparison files with progress bar
                for comparison_file_path in tqdm(comparison_files, desc=f"Comparing {subfolder_name}", unit="file", leave=False):
                    # Define relative path for embedding
                    # Relative to dataset folder
                    relative_dir = os.path.relpath(os.path.dirname(comparison_file_path), dataset_path)
                    relative_file = os.path.splitext(os.path.basename(comparison_file_path))[0]
                    comparison_relative_path = os.path.join(relative_dir, relative_file)

                    # Get or create embedding for the comparison code
                    embedding2 = get_embedding(comparison_file_path, embedding_base_path, comparison_relative_path, word2vec_model)
                    if embedding2 is None:
                        logging.warning(f"Failed to obtain embedding for '{comparison_file_path}'. Skipping similarity calculation.")
                        continue

                    # Collect embeddings and labels for visualization
                    all_embeddings.append(embedding2)
                    all_labels.append(subfolder_name)

                    similarity_ratio = calculate_similarity(embedding1, embedding2)
                    logging.info(f"Similarity between '{original_relative_path}' and '{comparison_relative_path}': {similarity_ratio:.4f}")

                    total_cases += 1

                    if subfolder_name == 'plagiarized':
                        if similarity_ratio >= SIMILARITY_THRESHOLD:
                            TP += 1  # Correctly identified as plagiarized
                        else:
                            FN += 1  # Plagiarized but not identified
                    elif subfolder_name == 'non-plagiarized':
                        if similarity_ratio < SIMILARITY_THRESHOLD:
                            TN += 1  # Correctly identified as non-plagiarized
                        else:
                            FP += 1  # Non-plagiarized but incorrectly identified as plagiarized

        # Calculate performance metrics for the current threshold
        if total_cases == 0:
            logging.warning(f"No cases found for threshold {SIMILARITY_THRESHOLD}. Skipping.\n")
            continue

        accuracy = (TP + TN) / total_cases
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Update the best threshold based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = SIMILARITY_THRESHOLD
            best_precision = precision
            best_recall = recall
            best_f_measure = f_measure
            best_TP = TP
            best_FP = FP
            best_FN = FN
            best_TN = TN

        # End timer for the current threshold
        threshold_end_time = time.time()
        elapsed = threshold_end_time - threshold_start_time

        # Log metrics for the current threshold
        logging.info(f"Threshold: {SIMILARITY_THRESHOLD}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F-measure: {f_measure:.4f}")
        logging.info(f"Processing Time for Threshold {SIMILARITY_THRESHOLD}: {elapsed:.2f} seconds")
        logging.info("-" * 40 + "\n")

    # After processing all thresholds, visualize embeddings
    if all_embeddings and all_labels:
        visualize_embeddings(all_embeddings, all_labels)

    # Plot the confusion matrix for the best threshold
    if best_threshold != 0:
        plot_confusion_matrix(best_TP, best_FP, best_FN, best_TN, best_threshold, save=True, save_path=f'confusion_matrix_{best_threshold}.png')

    # Print the best threshold and its corresponding metrics
    logging.info("=== Best Threshold Results ===")
    logging.info(f"Best Threshold: {best_threshold}")
    logging.info(f"Best Accuracy: {best_accuracy:.4f}")
    logging.info(f"Precision at Best Threshold: {best_precision:.4f}")
    logging.info(f"Recall at Best Threshold: {best_recall:.4f}")
    logging.info(f"F-measure at Best Threshold: {best_f_measure:.4f}")

    # End total execution timer
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    logging.info(f"Total Execution Time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
