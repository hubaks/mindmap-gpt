import streamlit as st
import markdown
from bs4 import BeautifulSoup
import networkx as nx
import json
from openai import OpenAI
import os
import tempfile
import pyvis.network as net



# Reuse the functions we defined earlier
def parse_markdown(content):
    return markdown.markdown(content)

def extract_mind_map(html):
    soup = BeautifulSoup(html, 'html.parser')
    mind_map = {}
    current_level = [mind_map]
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            while len(current_level) > level:
                current_level.pop()
            while len(current_level) < level:
                current_level.append({})
            current_level[-1][element.get_text()] = {}
            current_level.append(current_level[-1][element.get_text()])
        elif element.name == 'li':
            current_level[-1][element.get_text()] = {}
    
    return mind_map

def process_with_llm(mind_map):

    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    
    def process_node(node):
        for key, value in node.items():
            if isinstance(value, dict):
                prompt = "Given the parent node '{key}' and its child nodes {list(value.keys())}, describe the relationships between them. Be super concise and if your answer has spaces replace them with _ "
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[

                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}
                    ],
                )
                node[key] = {"content": value, "relationships": response.choices[0].message.content}
                process_node(value)
    
    process_node(mind_map)
    return mind_map

def extract_relationships(enhanced_mind_map):
    relationships = []
    
    def extract_from_node(node, parent=None):
        for key, value in node.items():
            if isinstance(value, dict) and "relationships" in value:
                relationships.append({
                    "source": parent,
                    "target": key,
                    "description": value["relationships"]
                })
                extract_from_node(value["content"], key)
            elif isinstance(value, dict):
                extract_from_node(value, key)
    
    extract_from_node(enhanced_mind_map)
    return relationships

def create_knowledge_graph(mind_map, relationships):
    G = nx.DiGraph()
    
    def add_nodes(data, parent=None):
        for key, value in data.items():
            G.add_node(key)
            if parent:
                G.add_edge(parent, key, type='hierarchical')
            if isinstance(value, dict):
                if "content" in value:
                    add_nodes(value["content"], key)
                else:
                    add_nodes(value, key)
    
    add_nodes(mind_map)
    
    for rel in relationships:
        if rel["source"] and rel["target"]:
            G.add_edge(rel["source"], rel["target"], type='semantic', description=rel["description"])
    
    return G

def visualize_graph(G):
    nt = net.Network(notebook=True, height="500px", width="100%")
    nt.from_nx(G)
    
    for edge in nt.edges:
        if edge['type'] == 'hierarchical':
            edge['color'] = 'blue'
        else:
            edge['color'] = 'red'
    
    return nt

# Streamlit app
st.title("Mind Map to Knowledge Graph")

uploaded_file = st.file_uploader("Choose a markdown file", type="md")

if uploaded_file is not None:
    content = uploaded_file.read().decode()
    
    with st.spinner("Processing..."):
        html = parse_markdown(content)
        mind_map = extract_mind_map(html)
        enhanced_mind_map = process_with_llm(mind_map)
        relationships = extract_relationships(enhanced_mind_map)
        knowledge_graph = create_knowledge_graph(mind_map, relationships)
    
    st.success("Processing complete!")
    
    st.subheader("Knowledge Graph Visualization")
    nt = visualize_graph(knowledge_graph)
    
    # Save and display the graph
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        nt.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            source_code = f.read()
    
    st.components.v1.html(source_code, height=600)
    
    st.subheader("Download Knowledge Graph")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gexf') as tmpfile:
        nx.write_gexf(knowledge_graph, tmpfile.name)
        with open(tmpfile.name, 'rb') as f:
            st.download_button(
                label="Download GEXF",
                data=f,
                file_name="knowledge_graph.gexf",
                mime="application/octet-stream"
            )

st.sidebar.info("Upload a markdown file to generate a knowledge graph.")