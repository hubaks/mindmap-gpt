import os
import numpy as np
import streamlit as st
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Set your OpenAI API Key
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()
# Neo4j connection details
url = "bolt://localhost:7687"
username = "neo4j"
password = "qwertyuiop"

# Initialize Neo4j driver and OpenAI embeddings model
driver = GraphDatabase.driver(url, auth=(username, password))
embeddings_model = OpenAIEmbeddings()

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

        # Find the most similar node
        most_similar_index = np.argmax(similarities)
        return nodes[most_similar_index]

def get_parents_and_children(node_id):
    with driver.session() as session:
        query = """
        MATCH (n)
        WHERE n.id = $id
        WITH n
        OPTIONAL MATCH (parent)-[:HAS_CHILD*0..]->(n)
        OPTIONAL MATCH (n)-[:HAS_CHILD*0..]->(child)
        RETURN DISTINCT parent.id AS ParentId, parent.content AS ParentContent,
                        n.id AS NodeId, n.content AS NodeContent,
                        child.id AS ChildId, child.content AS ChildContent
        """
        result = session.run(query, id=node_id)
        
        parents = {}
        children = {}
        node_content = None

        for record in result:
            if record["ParentId"]:
                parent_id = record["ParentId"]
                parent_content = record["ParentContent"] or "No content"
                parents[parent_id] = parent_content
            if record["ChildId"]:
                child_id = record["ChildId"]
                child_content = record["ChildContent"] or "No content"
                children[child_id] = child_content
            if record["NodeId"] == node_id:
                node_content = record["NodeContent"] or "No content"
        
        return node_content, parents, children

def format_context_for_api(nodes):
    """
    Format nodes into a string to be used as context for the OpenAI API.
    
    Args:
        nodes (dict): Dictionary of sorted nodes where keys are IDs and values are content.
        
    Returns:
        str: Formatted context string.
    """
    context_str = "The following context is part of a mind map:\n\n"
    
    for key, value in nodes.items():
        context_str += f"Node ID: {key}, Content: {value}\n"
    
    return context_str

def query_openai_api(context, user_query):
    """
    Send a query to the OpenAI API with the provided context.
    
    Args:
        context (str): Context string to provide background information.
        user_query (str): User's query to the chatbot.
        
    Returns:
        str: The response from the OpenAI API.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Specify the model you are using
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": user_query}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def query_and_retrieve_content(query_text):
    # Generate query embedding
    query_embedding = embeddings_model.embed_documents([query_text])
    if not query_embedding:
        raise ValueError("Failed to generate embedding for the query.")
    query_embedding = query_embedding[0]

    # Find the most similar node
    most_similar_node_id = find_most_similar_node(query_embedding)

    # Retrieve parent and child nodes along with the node content
    node_content, parents, children = get_parents_and_children(most_similar_node_id)

    # Combine results
    combined_content = {most_similar_node_id: node_content}
    combined_content.update(parents)  # Update with parents
    combined_content.update(children)  # Update with children

    # Convert IDs to a sortable format with integer conversion
    combined_content_int = {
        int(key): value for key, value in combined_content.items()
    }

    # Sort nodes by their ID in ascending order
    sorted_combined_content = dict(sorted(combined_content_int.items()))

    # Format context for the OpenAI API
    context = format_context_for_api(sorted_combined_content)

    # Get response from OpenAI API
    response = query_openai_api(context, query_text)

    return response

# Streamlit Interface
def main():
    st.title("Mind Map Query Interface")

    query_text = st.text_input("Enter your query:", "")

    if st.button("Submit"):
        if query_text:
            try:
                response = query_and_retrieve_content(query_text)
                st.subheader("OpenAI API Response:")
                st.write(response)
            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a query before submitting.")

# Run Streamlit app
if __name__ == "__main__":
    main()

# Close the driver
driver.close()
