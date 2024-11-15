import os
import javalang
import networkx as nx
import torch
import time
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from tqdm import tqdm
from datetime import datetime
import pickle  # For saving embeddings as serialized objects

# Initialize BERT model and tokenizer
print("Loading BERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}.")

# Create directory structure for embeddings
embeddings_dir = "embeddings"
bert_dir = os.path.join(embeddings_dir, "BERT")
plagiarized_dir = os.path.join(bert_dir, "plagiarized")
non_plagiarized_dir = os.path.join(bert_dir, "non-plagiarized")

# Create directories if they do not exist
os.makedirs(plagiarized_dir, exist_ok=True)
os.makedirs(non_plagiarized_dir, exist_ok=True)

def generate_filename(subfolder_name, java_file_name):
    """
    Generates a filename with the current datetime and BERT identifier.
    
    Args:
        subfolder_name (str): 'plagiarized' or 'non-plagiarized'
        java_file_name (str): Original Java file name
    
    Returns:
        str: Generated filename with datetime and BERT.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(java_file_name)[0]  # Remove .java extension
    filename = f"{base_name}_{current_time}_BERT.pkl"
    return os.path.join(bert_dir, subfolder_name, filename)

# Generate Program Dependence Graph (PDG) for Java code snippets
def generate_pdg(code):
    tokens = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()

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
                        # Ensure that the expression has a 'value' attribute
                        expr_value = getattr(stmt.expression, 'value', 'condition')
                        graph.add_edge(expr_value, method_name)

    return graph

# Convert PDG to BERT embedding
def pdg_to_bert_embedding(pdg):
    # Convert PDG to a string representation
    pdg_str = " ".join([f"{u}->{v}" for u, v in pdg.edges()])
    
    # Tokenize the PDG string
    tokens = tokenizer.tokenize(pdg_str)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([token_ids]).to(device)
    
    # Get BERT embedding
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[0]
    
    # Use the [CLS] token embedding as the PDG embedding
    pdg_embedding = embeddings[0].cpu().numpy()
    
    return pdg_embedding

# Define the path to the IR-Plag-Dataset folder
dataset_path = "D:/MyITSAcademia2-Season1/RPL/code_repository/codesim-gitor-o1/IR-Plag-Dataset"

# Define a list of similarity thresholds to iterate over
similarity_thresholds = [0.1, 0.2, 0.3]

# Initialize variables to keep track of the best result
best_threshold = 0
best_accuracy = 0

# Initialize counters
TP = 0
FP = 0
FN = 0

# Start the total execution timer
total_start_time = time.time()

# Loop through each similarity threshold and calculate accuracy
for SIMILARITY_THRESHOLD in tqdm(similarity_thresholds, desc="Processing Thresholds", unit="threshold"):
    threshold_start_time = time.time()
    
    # Initialize the counters for the current threshold
    total_cases = 0
    over_threshold_cases_plagiarized = 0
    over_threshold_cases_non_plagiarized = 0
    cases_plag = 0
    cases_non_plag = 0

    # Get list of folders once to avoid repeated disk access
    folder_names = os.listdir(dataset_path)
    
    # Loop through each subfolder in the dataset with a progress bar
    for folder_name in tqdm(folder_names, desc=f"Processing Folders for Threshold {SIMILARITY_THRESHOLD}", unit="folder", leave=False):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            # Find the Java file in the original folder
            original_path = os.path.join(folder_path, 'original')
            try:
                java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            except FileNotFoundError:
                print(f"Warning: 'original' folder not found in {folder_path}. Skipping.")
                continue

            if len(java_files) == 1:
                java_file = java_files[0]
                try:
                    with open(os.path.join(original_path, java_file), 'r', encoding='utf-8') as f:
                        code1 = f.read()
                except Exception as e:
                    print(f"Error reading {java_file}: {e}")
                    continue

                # Loop through each subfolder in the plagiarized and non-plagiarized folders
                for subfolder_name in ['plagiarized', 'non-plagiarized']:
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        # Get list of Java files with a progress bar
                        java_file_paths = []
                        for root, dirs, files in os.walk(subfolder_path):
                            for java_file in files:
                                if java_file.endswith('.java'):
                                    java_file_paths.append(os.path.join(root, java_file))
                        
                        for java_file_path in tqdm(java_file_paths, desc=f"Processing {subfolder_name} files", unit="file", leave=False):
                            try:
                                with open(java_file_path, 'r', encoding='utf-8') as f:
                                    code2 = f.read()
                            except Exception as e:
                                print(f"Error reading {java_file_path}: {e}")
                                continue

                            # Calculate the similarity ratio
                            try:
                                pdg_1 = generate_pdg(code1)
                                pdg_2 = generate_pdg(code2)
                                
                                # Convert PDGs to BERT embeddings
                                embedding_1 = pdg_to_bert_embedding(pdg_1)
                                embedding_2 = pdg_to_bert_embedding(pdg_2)
                                
                                # Save embeddings
                                # Determine which subdirectory to save based on 'plagiarized' or 'non-plagiarized'
                                subfolder_save_dir = plagiarized_dir if subfolder_name == 'plagiarized' else non_plagiarized_dir
                                
                                # Extract Java file name
                                java_file_name = os.path.basename(java_file_path)
                                
                                # Generate filenames
                                embedding1_filename = generate_filename(subfolder_name, f"{os.path.splitext(java_file_name)[0]}_1")
                                embedding2_filename = generate_filename(subfolder_name, f"{os.path.splitext(java_file_name)[0]}_2")
                                
                                # Save embedding1
                                with open(embedding1_filename, 'wb') as ef1:
                                    pickle.dump(embedding_1, ef1)
                                
                                # Save embedding2
                                with open(embedding2_filename, 'wb') as ef2:
                                    pickle.dump(embedding_2, ef2)
                                
                                # Calculate cosine similarity
                                similarity_ratio = 1 - cosine(embedding_1, embedding_2)
                            except Exception as e:
                                print(f"Error processing PDGs for {java_file_path}: {e}")
                                similarity_ratio = 0
                            
                            # Update the counters based on the similarity ratio
                            if subfolder_name == 'plagiarized':
                                cases_plag += 1
                                if similarity_ratio >= SIMILARITY_THRESHOLD:
                                    over_threshold_cases_plagiarized += 1
                                    TP += 1  # True positive
                                else:
                                    FN += 1  # False negative
                            elif subfolder_name == 'non-plagiarized':
                                cases_non_plag += 1
                                if similarity_ratio <= SIMILARITY_THRESHOLD:
                                    over_threshold_cases_non_plagiarized += 1
                                else:
                                    FP += 1  # False positive
                            total_cases += 1
            else:
                print(f"Error: Found {len(java_files)} Java files in {original_path} for {folder_name}. Expected exactly 1.")
    
    # Calculate accuracy for the current threshold
    if total_cases > 0:
        accuracy = (over_threshold_cases_non_plagiarized + over_threshold_cases_plagiarized) / total_cases
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = SIMILARITY_THRESHOLD

    # Calculate precision and recall
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # Calculate F-measure
    if precision + recall > 0:
        f_measure = 2 * (precision * recall) / (precision + recall)
    else:
        f_measure = 0

    # Calculate and print the time taken for the current threshold
    threshold_end_time = time.time()
    threshold_time = threshold_end_time - threshold_start_time
    print(f"\nThreshold {SIMILARITY_THRESHOLD}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F-measure={f_measure:.2f}")
    print(f"Time taken for threshold {SIMILARITY_THRESHOLD}: {threshold_time:.2f} seconds\n")

# Calculate total execution time
total_end_time = time.time()
total_time = total_end_time - total_start_time

# Print the best threshold and metrics
print(f"\n=== Final Results ===")
print(f"The best threshold is {best_threshold} with an accuracy of {best_accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F-measure: {f_measure:.2f}")
print(f"Total execution time: {total_time:.2f} seconds")
