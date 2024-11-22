import os
import io
import json
import jsonpickle
from transformers import AutoTokenizer, AutoModel  # No change needed
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import javalang 
import time  
from tqdm import tqdm 
from datetime import datetime 

# Load GraphCodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")  # Updated tokenizer
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")          # Updated model
model.eval()  # Set model to evaluation mode

def get_graphcodebert_embedding(code_string):  # Updated function name
    """
    Generates a GraphCodeBERT embedding for the given code string.

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

# Modified: Save embeddings in JSON format
def save_embedding_json(embedding, save_path):
    """
    Saves the embedding to the specified path in JSON format.

    Args:
        embedding (numpy.ndarray): The embedding vector to save.
        save_path (str): The file path where the embedding will be saved.
    """
    try:
        embedding_list = embedding.tolist()  # Convert to list for JSON serialization
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_list, f)
    except Exception as e:
        print(f"Error saving embedding to '{save_path}': {e}")

# Alternatively, keep using NumPy's .npy format for efficiency
def save_embedding_npy(embedding, save_path):
    """
    Saves the embedding to the specified path in NumPy's .npy format.

    Args:
        embedding (numpy.ndarray): The embedding vector to save.
        save_path (str): The file path where the embedding will be saved.
    """
    try:
        np.save(save_path, embedding)
    except Exception as e:
        print(f"Error saving embedding to '{save_path}': {e}")

def load_embedding_json(load_path):
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
        embedding = np.array(embedding_list)
        return embedding
    except Exception as e:
        print(f"Error loading embedding from '{load_path}': {e}")
        return None

def load_embedding_npy(load_path):
    """
    Loads the embedding from the specified .npy path.

    Args:
        load_path (str): The file path from where the embedding will be loaded.

    Returns:
        numpy.ndarray: The loaded embedding vector.
    """
    try:
        return np.load(load_path)
    except Exception as e:
        print(f"Error loading embedding from '{load_path}': {e}")
        return None

def ensure_directory(path):
    """
    Ensures that the directory exists; if not, creates it.

    Args:
        path (str): The directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)

def parse_java_code(file_path):
    """
    Parses Java code into an AST.

    Args:
        file_path (str): The path to the Java code file.

    Returns:
        javalang.tree.CompilationUnit: The AST of the Java code.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = javalang.parse.parse(code)
        return tree
    except javalang.parser.JavaSyntaxError as e:
        print(f"Syntax error while parsing '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"Error parsing file '{file_path}': {e}")
        return None

def save_ast(ast, save_path):
    """
    Saves the AST to the specified path in JSON format.

    Args:
        ast (javalang.tree.CompilationUnit): The AST to save.
        save_path (str): The file path where the AST will be saved.
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            # Serialize the AST using jsonpickle
            f.write(jsonpickle.encode(ast))
    except Exception as e:
        print(f"Error saving AST to '{save_path}': {e}")

def load_ast(load_path):
    """
    Loads the AST from the specified path.

    Args:
        load_path (str): The file path from where the AST will be loaded.

    Returns:
        javalang.tree.CompilationUnit: The loaded AST.
    """
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            ast = jsonpickle.decode(f.read())
        return ast
    except Exception as e:
        print(f"Error loading AST from '{load_path}': {e}")
        return None

def get_ast(file_path, ast_base_path, relative_path):
    """
    Retrieves the AST for a given file. If the AST does not exist, it generates and saves it.

    Args:
        file_path (str): The path to the Java code file.
        ast_base_path (str): The base directory where ASTs are stored.
        relative_path (str): The relative path within the ASTs directory corresponding to the code file.

    Returns:
        javalang.tree.CompilationUnit: The AST of the Java code.
    """
    ast_path = os.path.join(ast_base_path, relative_path + ".json")
    ast_dir = os.path.dirname(ast_path)
    ensure_directory(ast_dir)

    if os.path.exists(ast_path):
        # Load existing AST
        ast = load_ast(ast_path)
    else:
        # Parse the code file to generate AST
        ast = parse_java_code(file_path)
        if ast is not None:
            # Save AST
            save_ast(ast, ast_path)
    return ast

def get_embedding(file_path, embedding_base_path, relative_path, ast_base_path, ast_relative_path, use_json=True):
    """
    Retrieves the embedding for a given file. If the embedding does not exist, it generates and saves it.
    Also ensures that the AST exists and is saved.

    Args:
        file_path (str): The path to the code file.
        embedding_base_path (str): The base directory where embeddings are stored.
        relative_path (str): The relative path within the embeddings directory corresponding to the code file.
        ast_base_path (str): The base directory where ASTs are stored.
        ast_relative_path (str): The relative path within the ASTs directory corresponding to the code file.
        use_json (bool): Whether to save embeddings in JSON format. If False, saves as .npy.

    Returns:
        numpy.ndarray: The embedding vector.
    """
    # Ensure AST exists
    ast = get_ast(file_path, ast_base_path, ast_relative_path)
    if ast is None:
        print(f"Failed to obtain AST for '{file_path}'. Skipping embedding generation.")
        return None

    # Proceed to get or create embedding
    if use_json:
        embedding_filename = f"{relative_path}.json"
        save_embedding_func = save_embedding_json
        load_embedding_func = load_embedding_json
    else:
        embedding_filename = f"{relative_path}.npy"
        save_embedding_func = save_embedding_npy
        load_embedding_func = load_embedding_npy

    embedding_path = os.path.join(embedding_base_path, embedding_filename)
    embedding_dir = os.path.dirname(embedding_path)
    ensure_directory(embedding_dir)

    if os.path.exists(embedding_path):
        # Load existing embedding
        embedding = load_embedding_func(embedding_path)
        if embedding is None:
            print(f"Failed to load embedding from '{embedding_path}'.")
    else:
        # Read the code file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None
        # Generate embedding
        embedding = get_graphcodebert_embedding(code)  # Updated function call
        # Save embedding
        save_embedding_func(embedding, embedding_path)
    return embedding

def main():
    # Start total execution timer
    total_start_time = time.time()

    # Generate current datetime string in YYYYMMDD_HHMMSS format
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the path to the IR-Plag-Dataset folder
    dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")

    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
        return

    # Define the base path for embeddings and ASTs with timestamp
    # Updated path to reflect GraphCodeBERT
    embedding_base_path = os.path.join(os.getcwd(), "embeddings", "AST", "GraphCodeBERT", current_datetime)
    ast_base_path = os.path.join(os.getcwd(), "asts", "ASTs", current_datetime)
    ensure_directory(embedding_base_path)
    ensure_directory(ast_base_path)

    # Define a list of similarity thresholds to iterate over
    similarity_thresholds = [0.1, 0.5, 0.52]

    # Initialize variables to keep track of the best result
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f_measure = 0

    # Get list of folders to iterate with progress bar
    folder_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    # Loop through each similarity threshold and calculate accuracy
    for SIMILARITY_THRESHOLD in tqdm(similarity_thresholds, desc="Processing Thresholds", unit="threshold"):
        print(f"\nProcessing for Similarity Threshold: {SIMILARITY_THRESHOLD}")
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
                print(f"Original folder missing in '{folder_path}'. Skipping.")
                continue

            java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            if len(java_files) != 1:
                print(f"Error: Found {len(java_files)} Java files in '{original_path}' for '{folder_name}'. Expected exactly 1.")
                continue

            java_file = java_files[0]
            original_file_path = os.path.join(original_path, java_file)

            # Define relative paths for embedding and AST
            relative_file_name = os.path.splitext(java_file)[0]
            original_relative_path = os.path.join(folder_name, 'original', relative_file_name)
            original_ast_relative_path = os.path.join(folder_name, 'original', relative_file_name)

            # Get or create embedding for the original code
            embedding1 = get_embedding(
                file_path=original_file_path,
                embedding_base_path=embedding_base_path,
                relative_path=original_relative_path,
                ast_base_path=ast_base_path,
                ast_relative_path=original_ast_relative_path,
                use_json=True  # Set to False to use .npy format
            )
            if embedding1 is None:
                print(f"Failed to obtain embedding for '{original_file_path}'. Skipping comparisons.")
                continue

            # Loop through 'plagiarized' and 'non-plagiarized' subfolders with progress bar
            for subfolder_name in ['plagiarized', 'non-plagiarized']:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    print(f"Subfolder '{subfolder_name}' missing in '{folder_path}'. Skipping.")
                    continue

                # Collect all Java files in the subfolder
                comparison_files = []
                for root_dir, dirs, files in os.walk(subfolder_path):
                    for java_file in files:
                        if java_file.endswith('.java'):
                            comparison_file_path = os.path.join(root_dir, java_file)
                            comparison_files.append(comparison_file_path)

                if not comparison_files:
                    print(f"No Java files found in '{subfolder_path}'. Skipping.")
                    continue

                # Process comparison files with progress bar
                for comparison_file_path in tqdm(comparison_files, desc=f"Comparing {subfolder_name}", unit="file", leave=False):
                    # Define relative paths for embedding and AST
                    relative_dir = os.path.relpath(os.path.dirname(comparison_file_path), dataset_path)
                    relative_file = os.path.splitext(os.path.basename(comparison_file_path))[0]
                    comparison_relative_path = os.path.join(relative_dir, relative_file)
                    comparison_ast_relative_path = os.path.join(relative_dir, relative_file)

                    # Get or create embedding for the comparison code
                    embedding2 = get_embedding(
                        file_path=comparison_file_path,
                        embedding_base_path=embedding_base_path,
                        relative_path=comparison_relative_path,
                        ast_base_path=ast_base_path,
                        ast_relative_path=comparison_ast_relative_path,
                        use_json=True  # Set to False to use .npy format
                    )
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
            print(f"No cases found for threshold {SIMILARITY_THRESHOLD}. Skipping.\n")
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

        # End timer for the current threshold
        threshold_end_time = time.time()
        elapsed = threshold_end_time - threshold_start_time

        # Print metrics for the current threshold
        print(f"Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-measure: {f_measure:.4f}")
        print(f"Processing Time for Threshold {SIMILARITY_THRESHOLD}: {elapsed:.2f} seconds")
        print("-" * 40 + "\n")

    # Print the best threshold and its corresponding metrics
    print("=== Best Threshold Results ===")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Precision at Best Threshold: {best_precision:.4f}")
    print(f"Recall at Best Threshold: {best_recall:.4f}")
    print(f"F-measure at Best Threshold: {best_f_measure:.4f}")

    # End total execution timer
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"Total Execution Time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
