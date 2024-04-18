import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
import streamlit as st
def compoundVisual():
    def compound_to_graph(compound):
        try:
            # Convert SMILES string to RDKit molecule object
            mol = Chem.MolFromSmiles(compound)
            if mol is not None:
                # Generate molecular graph
                mol_graph = Chem.RWMol(mol)
                nx_graph = nx.Graph()

                # Add nodes for atoms
                for atom in mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    atom_symbol = atom.GetSymbol()
                    nx_graph.add_node(atom_idx, atom=atom_symbol)

                # Add edges for bonds
                for bond in mol.GetBonds():
                    start_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    bond_type = bond.GetBondTypeAsDouble()

                    # Check if an edge already exists between these atoms
                    if nx_graph.has_edge(start_idx, end_idx):
                        # Update bond type attribute
                        nx_graph[start_idx][end_idx]['bond_type'] = max(nx_graph[start_idx][end_idx]['bond_type'], bond_type)
                    else:
                        # Add a new edge with bond type attribute
                        nx_graph.add_edge(start_idx, end_idx, bond_type=bond_type)

                return nx_graph
            else:
                return None
        except Exception as e:
            print("Error:", e)
            return None

    # Example compound
    compound = "C1=CC=CC=C1CO"

    # Convert compound to graph
    compound_graph = compound_to_graph(compound)
    print("Graph nodes:", compound_graph.nodes)
    print("Graph edges:", compound_graph.edges)

    # Create a Matplotlib plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(compound_graph)  # Positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(compound_graph, pos, node_color='lightblue', node_size=700)

    # Draw edges
    edge_labels = {(u, v): f"{data['bond_type']}" for u, v, data in compound_graph.edges(data=True)}
    nx.draw_networkx_edges(compound_graph, pos, width=2, edge_color='gray')
    nx.draw_networkx_edge_labels(compound_graph, pos, edge_labels=edge_labels, font_size=10)

    # Draw labels (compound symbols)
    labels = {node: data['atom'] for node, data in compound_graph.nodes(data=True)}
    nx.draw_networkx_labels(compound_graph, pos, labels=labels, font_size=12, font_family="sans-serif")

    # Create legend for bond types
    single_bond_patch = plt.Line2D([], [], color='black', marker='_', markersize=10, label='Single Bond (1.0)')
    double_bond_patch = plt.Line2D([], [], color='black', marker='_', markersize=10, linestyle='dashed', label='Double Bond (2.0)')
    triple_bond_patch = plt.Line2D([], [], color='black', marker='_', markersize=10, linestyle='dotted', label='Triple Bond (3.0)')
    partial_double_bond_patch = plt.Line2D([], [], color='black', marker='_', markersize=10, linestyle='dashdot', label='Partial Double Bond (1.5)')
    plt.legend(handles=[single_bond_patch, double_bond_patch, triple_bond_patch, partial_double_bond_patch], loc='lower left')

    plt.title("Graph Representation of Compound :" +compound)
    plt.axis("off")

    # Display the Matplotlib plot in Streamlit
    st.pyplot(plt)

compoundVisual()