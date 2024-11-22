import os
import javalang
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nodevectors import ProNE

# Generate Program Dependence Graph (PDG) for Java code snippets
def generate_pdg(code):
    tokens = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except javalang.parser.JavaSyntaxError:
        # Handle cases where parsing fails
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
                        condition = getattr(stmt.expression, 'value', None)
                        if condition:
                            graph.add_edge(condition, method_name)

    return graph

# Generate graph embedding using ProNE
def get_pdg_embedding(graph, embed_dim=16):
    if graph.number_of_nodes() == 0:
        # Return a zero vector if the graph is empty
        return np.zeros(embed_dim)
    
    # Initialize ProNE
    proNE = ProNE(n_components=embed_dim)
    proNE.fit(graph)  # Fit ProNE on the graph

    embeddings = []
    for node in graph.nodes:
        node_embedding = proNE.predict(node)
        if len(node_embedding) > embed_dim:
            node_embedding = node_embedding[:embed_dim]  # Truncate if larger than embed_dim
        elif len(node_embedding) < embed_dim:
            node_embedding = np.pad(node_embedding, (0, embed_dim - len(node_embedding)), mode='constant')
        embeddings.append(node_embedding)

    # Aggregate node embeddings to form a graph embedding
    graph_embedding = np.mean(embeddings, axis=0)
    return graph_embedding

# Define the path to the IR-Plag-Dataset folder
dataset_path = os.path.join(os.getcwd(), "IR-Plag-Dataset")

# Define a list of similarity thresholds to iterate over
similarity_thresholds = [0.1, 0.2, 0.3]

# Initialize variables to keep track of the best result
best_threshold = 0
best_accuracy = 0

# Loop through each similarity threshold and calculate accuracy
for SIMILARITY_THRESHOLD in similarity_thresholds:
    # Initialize the counters for each threshold
    total_cases = 0
    over_threshold_cases_plagiarized = 0
    over_threshold_cases_non_plagiarized = 0
    cases_plag = 0
    cases_non_plag = 0

    # Initialize counters for metrics within the threshold
    TP = 0
    FP = 0
    FN = 0

    # Loop through each subfolder in the dataset
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            # Find the Java file in the original folder
            original_path = os.path.join(folder_path, 'original')
            if not os.path.exists(original_path):
                print(f"Error: 'original' subfolder does not exist in {folder_path}")
                continue

            java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
            if len(java_files) == 1:
                java_file = java_files[0]
                with open(os.path.join(original_path, java_file), 'r', encoding='utf-8') as f:
                    code1 = f.read()

                # Generate PDG and embedding for the original code
                pdg_1 = generate_pdg(code1)
                embedding_1 = get_pdg_embedding(pdg_1)

                # Loop through each subfolder in the plagiarized and non-plagiarized folders
                for subfolder_name in ['plagiarized', 'non-plagiarized']:
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        # Loop through each Java file in the subfolder
                        for root, dirs, files in os.walk(subfolder_path):
                            for java_file in files:
                                if java_file.endswith('.java'):
                                    with open(os.path.join(root, java_file), 'r', encoding='utf-8') as f:
                                        code2 = f.read()

                                    # Generate PDG and embedding for the compared code
                                    pdg_2 = generate_pdg(code2)
                                    embedding_2 = get_pdg_embedding(pdg_2)

                                    # Calculate cosine similarity between embeddings
                                    similarity_ratio = cosine_similarity(
                                        embedding_1.reshape(1, -1),
                                        embedding_2.reshape(1, -1)
                                    )[0][0]

                                    # Update counters based on similarity ratio and threshold
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
                print(f"Error: Found {len(java_files)} Java files in {original_path} for {folder_name}")

    # Calculate accuracy for the current threshold
    if total_cases > 0:
        accuracy = (over_threshold_cases_non_plagiarized + over_threshold_cases_plagiarized) / total_cases
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = SIMILARITY_THRESHOLD
    else:
        accuracy = 0

    # Calculate precision and recall for the current threshold
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # Calculate F-measure for the current threshold
    if precision + recall > 0:
        f_measure = 2 * (precision * recall) / (precision + recall)
    else:
        f_measure = 0

    # Print metrics for the current threshold
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F-measure: {f_measure:.2f}\n")

# Print the best threshold and accuracy after all iterations
print(f"The best threshold is {best_threshold} with an accuracy of {best_accuracy:.2f}")
