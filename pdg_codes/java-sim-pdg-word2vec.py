import os
import javalang
import networkx as nx
from gensim.models import Word2Vec  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  
from scipy.spatial.distance import cosine
import time  
from tqdm import tqdm  
from datetime import datetime  # Imported datetime module

# Function to generate Program Dependence Graph (PDG) for Java code snippets
def generate_pdg(code):
    """
    Generates a Program Dependence Graph (PDG) from Java code.

    Args:
        code (str): The Java code snippet.

    Returns:
        networkx.DiGraph: The generated PDG.
    """
    tokens = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except javalang.parser.JavaSyntaxError:
        # Handle parsing errors gracefully
        return nx.DiGraph()

    graph = nx.DiGraph()

    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_name = node.name
            graph.add_node(method_name)

            # Simulated data dependence
            for param in node.parameters:
                graph.add_edge(param.name, method_name)

            # Simulated control dependence
            if isinstance(node.body, javalang.tree.BlockStatement):
                for stmt in node.body.statements:
                    if isinstance(stmt, javalang.tree.IfStatement):
                        condition = getattr(stmt.expression, 'value', 'condition')
                        graph.add_edge(condition, method_name)

    return graph

# Function to preprocess PDG into tokens
def preprocess_pdg(pdg):
    """
    Preprocesses PDG by converting edges into tokenized strings.

    Args:
        pdg (networkx.DiGraph): The Program Dependence Graph.

    Returns:
        list: A list of tokens representing the PDG.
    """
    pdg_str = " ".join([f"{u}->{v}" for u, v in pdg.edges()])
    tokens = pdg_str.split()  # Simple tokenization; can be enhanced
    return tokens

# Function to get Word2Vec embedding by averaging word vectors
def get_word2vec_embedding(model, tokens):
    """
    Generates a Word2Vec embedding by averaging word vectors.

    Args:
        model (Word2Vec): Trained Word2Vec model.
        tokens (list): Tokenized PDG.

    Returns:
        numpy.ndarray: The embedding vector.
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Function to calculate cosine similarity between two embeddings
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

# Functions for saving and loading embeddings
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

# Function to get or create embedding using specified embedding type
def get_embedding(file_path, embedding_base_path, relative_path, word2vec_model, embedding_type='word2vec', run_id=''):
    """
    Retrieves the embedding for a given file. If the embedding does not exist, it generates and saves it.

    Args:
        file_path (str): The path to the code file.
        embedding_base_path (str): The base directory where embeddings are stored.
        relative_path (str): The relative path within the embeddings directory corresponding to the code file.
        word2vec_model (Word2Vec): Trained Word2Vec model.
        embedding_type (str): The type of embedding (e.g., 'word2vec').
        run_id (str): The identifier for the current run based on datetime.

    Returns:
        numpy.ndarray: The embedding vector.
    """
    # Construct the embedding directory based on embedding type and run_id
    embedding_dir = os.path.join(embedding_base_path, embedding_type, run_id, os.path.dirname(relative_path))
    ensure_directory(embedding_dir)
    
    # Incorporate run_id into the filename
    base_filename = os.path.basename(relative_path)
    embedding_filename = f"{base_filename}_{run_id}.npy"
    embedding_path = os.path.join(embedding_dir, embedding_filename)

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
        # Generate PDG
        pdg = generate_pdg(code)
        # Preprocess PDG into tokens
        tokens = preprocess_pdg(pdg)
        # Generate embedding based on type
        if embedding_type == 'word2vec':
            embedding = get_word2vec_embedding(word2vec_model, tokens)
        else:
            print(f"Embedding type '{embedding_type}' is not supported.")
            return None
        # Save embedding
        save_embedding(embedding, embedding_path)
    return embedding

def main():
    # Start total execution timer
    total_start_time = time.time()

    # Define the path to the IR-Plag-Dataset folder
    dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")

    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
        return

    # Generate a run identifier based on current datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    print(f"Run ID: {run_id}")

    # Define the base path for embeddings, incorporating run_id
    embedding_base_path = os.path.join(os.getcwd(), "embeddings")
    ensure_directory(embedding_base_path)

    # Specify the embedding type
    embedding_type = 'word2vec'  # You can change this to other types if implemented

    # Collect all Java files for training Word2Vec
    java_file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                java_file_paths.append(file_path)

    if not java_file_paths:
        print("No Java files found in the dataset to train Word2Vec model.")
        return

    # Preprocess all Java files with progress bar
    print("Preprocessing Java files for Word2Vec training...")
    all_tokens = []
    start_time = time.time()
    for file_path in tqdm(java_file_paths, desc="Preprocessing Files", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            pdg = generate_pdg(code)
            tokens = preprocess_pdg(pdg)
            all_tokens.append(tokens)
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.\n")

    # Train Word2Vec model
    print("Training Word2Vec model...")
    start_time = time.time()
    word2vec_model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)
    end_time = time.time()
    print(f"Word2Vec model training completed in {end_time - start_time:.2f} seconds.\n")

    # Save the Word2Vec model with run_id in the filename
    model_filename = f"word2vec_model_{run_id}.model"
    model_save_path = os.path.join(embedding_base_path, embedding_type, run_id, model_filename)
    ensure_directory(os.path.dirname(model_save_path))
    word2vec_model.save(model_save_path)
    print(f"Word2Vec model saved at '{model_save_path}'.\n")

    # Define a list of similarity thresholds to iterate over
    similarity_thresholds = [0.1, 0.2, 0.3]

    # Initialize variables to keep track of the best result
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f_measure = 0

    # Initialize counters
    TP_total = 0
    FP_total = 0
    FN_total = 0

    # Loop through each similarity threshold and calculate accuracy
    for SIMILARITY_THRESHOLD in similarity_thresholds:
        print(f"Processing for Similarity Threshold: {SIMILARITY_THRESHOLD}")
        # Start timer for each threshold
        threshold_start_time = time.time()

        # Initialize the counters for the current threshold
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives
        total_cases = 0

        # Get list of folders to iterate with progress bar
        folder_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

        for folder_name in tqdm(folder_names, desc="Processing Folders", unit="folder"):
            folder_path = os.path.join(dataset_path, folder_name)
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
            embedding1 = get_embedding(
                file_path=original_file_path,
                embedding_base_path=embedding_base_path,
                relative_path=original_relative_path,
                word2vec_model=word2vec_model,
                embedding_type=embedding_type,
                run_id=run_id  # Pass run_id
            )
            if embedding1 is None:
                print(f"Failed to obtain embedding for '{original_file_path}'. Skipping comparisons.")
                continue

            # Loop through 'plagiarized' and 'non-plagiarized' subfolders
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

                # Process comparison files with progress bar
                for comparison_file_path in tqdm(comparison_files, desc=f"Comparing {subfolder_name}", unit="file", leave=False):
                    # Define relative path for embedding
                    relative_dir = os.path.relpath(os.path.dirname(comparison_file_path), dataset_path)
                    relative_file = os.path.splitext(os.path.basename(comparison_file_path))[0]
                    comparison_relative_path = os.path.join(relative_dir, relative_file)

                    # Get or create embedding for the comparison code
                    embedding2 = get_embedding(
                        file_path=comparison_file_path,
                        embedding_base_path=embedding_base_path,
                        relative_path=comparison_relative_path,
                        word2vec_model=word2vec_model,
                        embedding_type=embedding_type,
                        run_id=run_id  # Pass run_id
                    )
                    if embedding2 is None:
                        print(f"Failed to obtain embedding for '{comparison_file_path}'. Skipping similarity calculation.")
                        continue

                    # Calculate cosine similarity
                    similarity_ratio = calculate_similarity(embedding1, embedding2)

                    # Update the counters based on the similarity ratio
                    if subfolder_name == 'plagiarized':
                        if similarity_ratio >= SIMILARITY_THRESHOLD:
                            TP += 1  # True positive: plagiarized and identified as plagiarized
                        else:
                            FN += 1  # False negative: plagiarized but identified as non-plagiarized
                    elif subfolder_name == 'non-plagiarized':
                        if similarity_ratio < SIMILARITY_THRESHOLD:
                            TN += 1  # True negative: non-plagiarized and identified as non-plagiarized
                        else:
                            FP += 1  # False positive: non-plagiarized but identified as plagiarized
                    total_cases += 1

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
