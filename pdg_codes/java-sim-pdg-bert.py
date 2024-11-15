import os
import javalang
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
                        graph.add_edge(stmt.expression.value, method_name)

    return graph

# Convert PDG to BERT embedding
def pdg_to_bert_embedding(pdg):
    # Convert PDG to a string representation
    pdg_str = " ".join([f"{u}->{v}" for u, v in pdg.edges()])
    
    # Tokenize the PDG string
    inputs = tokenizer(pdg_str, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get BERT embedding
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    # Use the [CLS] token embedding as the PDG embedding
    pdg_embedding = embeddings[:, 0, :].squeeze().cpu().numpy()
    
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

# Loop through each similarity threshold and calculate accuracy
for SIMILARITY_THRESHOLD in similarity_thresholds:
    # Initialize the counters
    total_cases = 0
    over_threshold_cases_plagiarized = 0
    over_threshold_cases_non_plagiarized = 0
    cases_plag = 0
    cases_non_plag = 0

    # Loop through each subfolder in the dataset
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            # Find the Java file in the original folder
            original_path = os.path.join(folder_path, 'original')
            java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            if len(java_files) == 1:
                java_file = java_files[0]
                with open(os.path.join(original_path, java_file), 'r') as f:
                    code1 = f.read()

                # Generate PDG and embedding for the original code
                try:
                    pdg_1 = generate_pdg(code1)
                    embedding_1 = pdg_to_bert_embedding(pdg_1)
                except Exception as e:
                    print(f"Error processing original file {java_file} in {folder_name}: {e}")
                    continue

                # Loop through each subfolder in the plagiarized and non-plagiarized folders
                for subfolder_name in ['plagiarized', 'non-plagiarized']:
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        # Loop through each Java file in the subfolder
                        for root, dirs, files in os.walk(subfolder_path):
                            for java_file in files:
                                if java_file.endswith('.java'):
                                    with open(os.path.join(root, java_file), 'r') as f:
                                        code2 = f.read()
                                    # Calculate the similarity ratio
                                    try:
                                        pdg_2 = generate_pdg(code2)
                                        embedding_2 = pdg_to_bert_embedding(pdg_2)
                                        
                                        # Calculate cosine similarity
                                        similarity_ratio = 1 - cosine(embedding_1, embedding_2)
                                    except Exception as e:
                                        print(f"Error processing file {java_file} in {subfolder_name} of {folder_name}: {e}")
                                        similarity_ratio = 0
                                    
                                    # Update the counters based on the similarity ratio
                                    if subfolder_name == 'plagiarized':
                                        cases_plag += 1
                                        if similarity_ratio >= SIMILARITY_THRESHOLD:
                                            over_threshold_cases_plagiarized += 1
                                            TP += 1  # True positive: plagiarized and identified as plagiarized
                                        else:
                                            FN += 1  # False negative: plagiarized but identified as non-plagiarized
                                    elif subfolder_name == 'non-plagiarized':
                                        cases_non_plag += 1
                                        if similarity_ratio <= SIMILARITY_THRESHOLD:
                                            over_threshold_cases_non_plagiarized += 1
                                        else:
                                            FP += 1  # False positive: non-plagiarized but identified as plagiarized
                                    total_cases += 1
            else:
                print(f"Error: Found {len(java_files)} Java files in {original_path} for {folder_name}")

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

# Print the best threshold and accuracy
print(f"The best threshold is {best_threshold} with an accuracy of {best_accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F-measure: {f_measure:.2f}")
