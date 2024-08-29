import os
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

# Set your OpenAI API Key
os.environ['OPENAI_API_KEY'] = ""

# Neo4j connection details
url = "bolt://localhost:7687"
username = "neo4j"
password = "qwertyuiop"

# Initialize Neo4j driver
driver = GraphDatabase.driver(url, auth=(username, password))

# Initialize OpenAI Embeddings model
embeddings_model = OpenAIEmbeddings()

# Function to generate and store embeddings
def generate_and_store_embeddings():
    with driver.session() as session:
        # Fetch all nodes, avoiding deprecated features
        result = session.run(
            "MATCH (n) RETURN n AS node_id, n.content AS content"
        )

        for record in result:
            node_id = record['node_id']
            text_data = record['content']
            
            # Generate embedding for the node's text data
            embedding = embeddings_model.embed_documents([text_data])[0]
            
            # Store the embedding back into the node
            session.run(
                "MATCH (n) WHERE id(n) = $node_id SET n.embedding = $embedding",
                node_id=node_id,
                embedding=embedding
            )
            print(f"Stored embedding for node {node_id}")

# Generate and store embeddings for all relevant nodes
generate_and_store_embeddings()

# Close the driver
driver.close()
