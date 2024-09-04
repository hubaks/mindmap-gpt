import streamlit as st
import plotly.graph_objects as go
import numpy as np
from neo4j import GraphDatabase
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import os 

# Set your OpenAI API Key
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()

# Neo4j connection details
url = "bolt://localhost:7687"
username = "neo4j"
password = ""

# Initialize Neo4j driver and OpenAI embeddings model
driver = GraphDatabase.driver(url, auth=(username, password))
embeddings_model = OpenAIEmbeddings()

@st.cache_data
def get_vector_from_db(node_id):
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.id = $id RETURN n.embedding AS embedding",
            id=node_id
        )
        record = result.single()
        if record:
            embedding = record["embedding"]
            if embedding is not None:
                return np.array(embedding)
        return None

@st.cache_data
def find_most_similar_node(query_embedding):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n.id AS id, n.embedding AS embedding")
        nodes = []
        embeddings = []
        for record in result:
            node_id = record["id"]
            embedding = record["embedding"]
            if embedding is not None:
                nodes.append(node_id)
                embeddings.append(np.array(embedding))
        if not embeddings:
            raise ValueError("No embeddings found in the database.")
        embeddings = np.array(embeddings)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        most_similar_index = np.argmax(similarities)
        return nodes[most_similar_index]

@st.cache_data
def get_parents_and_children(node_id):
    with driver.session() as session:
        query = """
        MATCH (n)
        WHERE n.id = $id
        OPTIONAL MATCH (parent)-[:HAS_CHILD]->(n)
        OPTIONAL MATCH (n)-[:HAS_CHILD]->(child)
        RETURN 
            COLLECT(DISTINCT {id: parent.id, content: parent.content}) AS parents,
            n.content AS node_content,
            COLLECT(DISTINCT {id: child.id, content: child.content}) AS children
        """
        result = session.run(query, id=node_id)
        record = result.single()
        parents = {parent["id"]: parent["content"] or "No content" for parent in record["parents"] if parent["id"]}
        children = {child["id"]: child["content"] or "No content" for child in record["children"] if child["id"]}
        node_content = record["node_content"] or "No content"
        return {"node_content": node_content, "parents": parents, "children": children}
    
def determine_intent(user_query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Determine if the user wants to add or edit a node in a mind map."},
            {"role": "user", "content": user_query}
        ],
        max_tokens=10,
        temperature=0.5
    )
    intent = response.choices[0].message.content.lower()
    if "add" in intent:
        return "add"
    elif "edit" in intent:
        return "edit"
    else:
        return None

def edit_node(node_id, new_content):
    with driver.session() as session:
        session.run(
            "MATCH (n) WHERE n.id = $node_id "
            "SET n.content = $new_content",
            node_id=node_id,
            new_content=new_content
        )

def add_node(parent_id, new_node_content):
    with driver.session() as session:
        session.run(
            "MATCH (parent) WHERE parent.id = $parent_id "
            "CREATE (child:Node {content: $new_node_content}) "
            "CREATE (parent)-[:HAS_CHILD]->(child)",
            parent_id=parent_id,
            new_node_content=new_node_content
        )

# Streamlit app
st.title("Interactive Graph Editor")

# User query input
user_query = st.text_input("Enter your query:")

if user_query:
        # Find the most similar node
    query_embedding = embeddings_model.embed_documents([user_query])[0]
    most_similar_node_id = find_most_similar_node(query_embedding)
    
    # Determine the user's intent
    intent = determine_intent(user_query)
    
    # Get parents and children of the most similar node
    related_nodes = get_parents_and_children(most_similar_node_id)
    
    # Display the subgraph
    node_content = related_nodes["node_content"]
    parents = related_nodes["parents"]
    children = related_nodes["children"]

    # Plot the subgraph
    labels = list(parents.values()) + [node_content] + list(children.values())
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=[labels.index(parent) for parent in parents.values()] + [labels.index(node_content)] * len(children),
            target=[labels.index(node_content)] * len(parents) + [labels.index(child) for child in children.values()],
            value=[1] * (len(parents) + len(children)),
        ),
    )])
    st.plotly_chart(fig)

    # Select a node to edit
    node_options = list(parents.values()) + [node_content] + list(children.values())
    node_ids = list(parents.keys()) + [most_similar_node_id] + list(children.keys())
    node_dict = dict(zip(node_options, node_ids))
    selected_node = st.selectbox("Select a node to edit:", node_options)
    selected_node_id = node_dict[selected_node]

    # Edit node content
    if intent == "edit":
        new_content = st.text_input("Edit Node Content:")
        if st.button("Update Node"):
            edit_node(selected_node_id, new_content)
            st.success(f"Node '{selected_node}' updated successfully.")

    # Add new node
    elif intent == "add":
        new_node_content = st.text_input("New Node Content:")
        if st.button("Add Node"):
            add_node(selected_node_id, new_node_content)
            st.success(f"Node '{new_node_content}' added successfully.")