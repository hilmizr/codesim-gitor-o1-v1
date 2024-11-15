import os
import io
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(code_string):
    """
    Generates a BERT embedding for the given code string.

    Args:
        code_string (str): The raw code snippet.

    Returns:
        numpy.ndarray: The embedding vector.
    """
    inputs = tokenizer(code_string, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as a representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

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

def save_embedding(embedding, save_path):
    """
    Saves the embedding to the specified path.

    Args:
        embedding (numpy.ndarray): The embedding vector to save.
        save_path (str): The file path where the embedding will be saved.
    """
    np.save(save_path, embedding)

def load_embedding(load_path):
    """
    Loads the embedding from the specified path.

    Args:
        load_path (str): The file path from where the embedding will be loaded.

    Returns:
        numpy.ndarray: The loaded embedding vector.
    """
    return np.load(load_path)

def ensure_directory(path):
    """
    Ensures that the directory exists; if not, creates it.

    Args:
        path (str): The directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)

def get_embedding(file_path, embedding_base_path, relative_path):
    """
    Retrieves the embedding for a given file. If the embedding does not exist, it generates and saves it.

    Args:
        file_path (str): The path to the code file.
        embedding_base_path (str): The base directory where embeddings are stored.
        relative_path (str): The relative path within the embeddings directory corresponding to the code file.

    Returns:
        numpy.ndarray: The embedding vector.
    """
    embedding_path = os.path.join(embedding_base_path, relative_path + ".npy")
    embedding_dir = os.path.dirname(embedding_path)
    ensure_directory(embedding_dir)
    
    if os.path.exists(embedding_path):
        # Load existing embedding
        embedding = load_embedding(embedding_path)
    else:
        # Read the code file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None
        # Generate embedding
        embedding = get_bert_embedding(code)
        # Save embedding
        save_embedding(embedding, embedding_path)
    return embedding

def main():
    # Define the path to the IR-Plag-Dataset folder
    dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
        return
    
    # Define the base path for embeddings
    embedding_base_path = os.path.join(os.getcwd(), "embeddings")
    ensure_directory(embedding_base_path)
    
    # Define a list of similarity thresholds to iterate over
    similarity_thresholds = [0.1, 0.5, 0.52]
    
    # Initialize variables to keep track of the best result
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f_measure = 0
    
    # Loop through each similarity threshold and calculate accuracy
    for SIMILARITY_THRESHOLD in similarity_thresholds:
        # Initialize the counters for the current threshold
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives
        total_cases = 0
        
        # Loop through each subfolder in the dataset
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.isdir(folder_path):
                # Path to the original Java file
                original_path = os.path.join(folder_path, 'original')
                if not os.path.exists(original_path):
                    print(f"Original folder missing in '{folder_path}'. Skipping.")
                    continue
                
                java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
                if len(java_files) != 1:
                    print(f"Error: Found {len(java_files)} Java files in '{original_path}' for '{folder_name}'. Expected exactly 1.")
                    continue
                
                java_file = java_files[0]
                original_file_path = os.path.join(original_path, java_file)
                
                # Define relative path for embedding
                original_relative_path = os.path.join(folder_name, 'original', os.path.splitext(java_file)[0])
                
                # Get or create embedding for the original code
                embedding1 = get_embedding(original_file_path, embedding_base_path, original_relative_path)
                if embedding1 is None:
                    print(f"Failed to obtain embedding for '{original_file_path}'. Skipping comparisons.")
                    continue
                
                # Loop through 'plagiarized' and 'non-plagiarized' subfolders
                for subfolder_name in ['plagiarized', 'non-plagiarized']:
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if not os.path.isdir(subfolder_path):
                        print(f"Subfolder '{subfolder_name}' missing in '{folder_path}'. Skipping.")
                        continue
                    
                    # Walk through all Java files in the subfolder
                    for root, dirs, files in os.walk(subfolder_path):
                        for java_file in files:
                            if java_file.endswith('.java'):
                                comparison_file_path = os.path.join(root, java_file)
                                
                                # Define relative path for embedding
                                # Relative to dataset folder
                                relative_dir = os.path.relpath(root, dataset_path)
                                relative_file = os.path.splitext(java_file)[0]
                                comparison_relative_path = os.path.join(relative_dir, relative_file)
                                
                                # Get or create embedding for the comparison code
                                embedding2 = get_embedding(comparison_file_path, embedding_base_path, comparison_relative_path)
                                if embedding2 is None:
                                    print(f"Failed to obtain embedding for '{comparison_file_path}'. Skipping similarity calculation.")
                                    continue
                                
                                similarity_ratio = calculate_similarity(embedding1, embedding2)
                                # print(f"{subfolder_name},{similarity_ratio:.2f}")  # Uncomment for detailed logs
                                
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
            print(f"No cases found for threshold {SIMILARITY_THRESHOLD}. Skipping.")
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
        
        # Print metrics for the current threshold
        print(f"Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-measure: {f_measure:.4f}")
        print("-" * 40)
    
    # Print the best threshold and its corresponding metrics
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Precision at Best Threshold: {best_precision:.4f}")
    print(f"Recall at Best Threshold: {best_recall:.4f}")
    print(f"F-measure at Best Threshold: {best_f_measure:.4f}")

if __name__ == "__main__":
    main()
