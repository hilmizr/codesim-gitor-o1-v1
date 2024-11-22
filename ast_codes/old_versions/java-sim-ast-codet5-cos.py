import os
import javalang
from transformers import T5Tokenizer, T5EncoderModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Ensure reproducibility and manage device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CodeT5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-base")
model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
model.to(device)
model.eval()

def get_ast(code):
    """
    Parses Java code into its Abstract Syntax Tree (AST).
    
    Args:
        code (str): Java source code.
    
    Returns:
        javalang.ast.Node: Root of the AST.
    """
    try:
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        return parser.parse()
    except javalang.parser.JavaSyntaxError as e:
        print(f"Java syntax error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during AST parsing: {e}")
        return None

def ast_to_string(ast):
    """
    Converts AST to a single-line string representation.
    
    Args:
        ast (javalang.ast.Node): Abstract Syntax Tree.
    
    Returns:
        str: String representation of the AST.
    """
    if ast is None:
        return ""
    return str(ast).replace('\n', ' ').replace('\r', ' ').strip()

def get_codet5_embedding(ast_string):
    """
    Generates an embedding for the given AST string using CodeT5.
    
    Args:
        ast_string (str): String representation of the AST.
    
    Returns:
        numpy.ndarray: Embedding vector.
    """
    if not ast_string:
        return None
    inputs = tokenizer(
        ast_string,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over the sequence length
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def calculate_similarity(snippet1, snippet2):
    """
    Calculates cosine similarity between two code snippets based on their AST embeddings.
    
    Args:
        snippet1 (str): First Java code snippet.
        snippet2 (str): Second Java code snippet.
    
    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    # Get the ASTs for the code snippets
    ast1 = get_ast(snippet1)
    ast2 = get_ast(snippet2)

    # Convert ASTs to string representation
    ast_str1 = ast_to_string(ast1)
    ast_str2 = ast_to_string(ast2)

    # Get CodeT5 embeddings for ASTs
    embedding1 = get_codet5_embedding(ast_str1)
    embedding2 = get_codet5_embedding(ast_str2)

    if embedding1 is None or embedding2 is None:
        return 0.0  # If embedding failed, treat similarity as 0

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )[0][0]

    return similarity

def main():
    """
    Main function to evaluate plagiarism detection using CodeT5 embeddings.
    """
    # Define the path to the IR-Plag-Dataset folder
    dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")

    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist.")
        return

    # Define a list of similarity thresholds to iterate over
    similarity_thresholds = [0.1, 0.5, 0.52]

    # Initialize variables to keep track of the best result
    best_threshold = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f_measure = 0

    # Loop through each similarity threshold and calculate metrics
    for SIMILARITY_THRESHOLD in similarity_thresholds:
        # Initialize the counters
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives
        total_cases = 0

        # Loop through each subfolder in the dataset
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue  # Skip if not a directory

            # Define paths for original, plagiarized, and non-plagiarized folders
            original_path = os.path.join(folder_path, 'original')
            plagiarized_path = os.path.join(folder_path, 'plagiarized')
            non_plagiarized_path = os.path.join(folder_path, 'non-plagiarized')

            # Check existence of original folder
            if not os.path.isdir(original_path):
                print(f"Original folder missing in '{folder_name}'. Skipping...")
                continue

            # Find the Java file in the original folder
            java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            if len(java_files) != 1:
                print(f"Error: Found {len(java_files)} Java files in '{original_path}' for '{folder_name}'. Expected exactly 1.")
                continue

            java_file = java_files[0]
            original_file_path = os.path.join(original_path, java_file)
            try:
                with open(original_file_path, 'r', encoding='utf-8') as f:
                    code1 = f.read()
            except Exception as e:
                print(f"Error reading '{original_file_path}': {e}")
                continue

            # Define pairs of subfolders and their labels
            comparison_pairs = [
                (plagiarized_path, 'plagiarized'),
                (non_plagiarized_path, 'non-plagiarized')
            ]

            for subfolder_path, label in comparison_pairs:
                if not os.path.isdir(subfolder_path):
                    print(f"Subfolder '{label}' missing in '{folder_name}'. Skipping...")
                    continue

                # Traverse through all Java files in the subfolder
                for root, dirs, files in os.walk(subfolder_path):
                    for java_file in files:
                        if not java_file.endswith('.java'):
                            continue
                        suspect_file_path = os.path.join(root, java_file)
                        try:
                            with open(suspect_file_path, 'r', encoding='utf-8') as f:
                                code2 = f.read()
                        except Exception as e:
                            print(f"Error reading '{suspect_file_path}': {e}")
                            continue

                        similarity_ratio = calculate_similarity(code1, code2)

                        # Update counters based on label and similarity
                        if label == 'plagiarized':
                            if similarity_ratio >= SIMILARITY_THRESHOLD:
                                TP += 1
                            else:
                                FN += 1
                        elif label == 'non-plagiarized':
                            if similarity_ratio >= SIMILARITY_THRESHOLD:
                                FP += 1
                            else:
                                TN += 1

                        total_cases += 1

        # Calculate evaluation metrics
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0

        if precision + recall > 0:
            f_measure = 2 * (precision * recall) / (precision + recall)
        else:
            f_measure = 0.0

        accuracy = (TP + TN) / total_cases if total_cases > 0 else 0.0

        # Update best threshold if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = SIMILARITY_THRESHOLD
            best_precision = precision
            best_recall = recall
            best_f_measure = f_measure

        # Optional: Print intermediate results for each threshold
        print(f"Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-measure: {f_measure:.4f}\n")

    # Final Output
    print("========================================")
    print("Best Threshold and Evaluation Metrics:")
    print(f"Threshold: {best_threshold}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F-measure: {best_f_measure:.4f}")
    print("========================================")

if __name__ == "__main__":
    main()
