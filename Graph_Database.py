import streamlit as st
from neo4j import GraphDatabase

# Parsing the Markdown file into nodes and parent-child relationships
def parse_mind_map(md_content):
    nodes = []
    stack = []
    for line in md_content.splitlines():
        indent_level = len(line) - len(line.lstrip())
        node_content = line.strip()
        
        # Adjust the stack to match the current indentation level
        while len(stack) > indent_level:
            stack.pop()
        
        node_id = len(nodes)
        nodes.append((node_content, node_id))
        
        # Add the current node to the stack
        stack.append(node_id)
    
    # Create the children dictionary
    children = {}
    for i, (content, node_id) in enumerate(nodes):
        parent_id = None
        if i > 0:
            parent_id = nodes[i-1][1]
        children[node_id] = []
        if parent_id is not None:
            children[parent_id].append(node_id)
    
    return nodes, children

# Insert nodes into Neo4j
def insert_mind_map_into_neo4j(nodes):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", ""))
    with driver.session() as session:
        for node_id, (content, _) in enumerate(nodes):
            session.write_transaction(create_node, node_id, content)
            if node_id > 0:
                session.write_transaction(create_relationship, node_id - 1, node_id, "HAS_CHILD")
    driver.close()

# Create a node in Neo4j
def create_node(tx, node_id, content):
    tx.run("CREATE (n:Node {id: $id, content: $content})", id=node_id, content=content)

# Create a relationship in Neo4j
def create_relationship(tx, parent_id, child_id, relationship_type):
    query = f"MATCH (a:Node {{id: {parent_id}}}), (b:Node {{id: {child_id}}}) CREATE (a)-[:{relationship_type}]->(b)"
    tx.run(query)

# Extract the mind map from Neo4j and rebuild the Markdown
def extract_mind_map_from_neo4j():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
    with driver.session() as session:
        result = session.run("MATCH (n:Node) OPTIONAL MATCH (n)-[:HAS_CHILD]->(m) RETURN n, collect(m) AS children")
        nodes = {}
        children = {}
        for record in result:
            node = record["n"]
            child_nodes = record["children"]
            if node is not None:
                node_id = node["id"]
                nodes[node_id] = (node["content"], node["id"])
                child_ids = [child["id"] for child in child_nodes if child is not None]
                children[node_id] = child_ids
    driver.close()
    return nodes, children

# Rebuild the Markdown structure
def rebuild_markdown(nodes, children):
    print("Nodes:", nodes)
    print("Children:", children)

    def build_markdown(node_id, indent=0):
        content, _ = nodes[node_id]
        md_lines = [" " * indent + content]
        if node_id in children:
            for child_id in sorted(children[node_id]):
                md_lines.extend(build_markdown(child_id, indent + 2))
        return md_lines

    root_nodes = [node_id for node_id, child_ids in children.items() if not any(node_id in child_ids for _, child_ids in children.items())]
    print(root_nodes)
    markdown_lines = []
    for root_node in root_nodes:
        markdown_lines.extend(build_markdown(root_node))
    
    return "\n".join(markdown_lines)


# Streamlit app
st.title("Mind Map Graph Database POC")

uploaded_file = st.file_uploader("Upload a Markdown file", type="md")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    nodes, children = parse_mind_map(content)
    insert_mind_map_into_neo4j(nodes)

    st.write("Original Markdown:")
    st.code(content)

    extracted_nodes, extracted_children = extract_mind_map_from_neo4j()
    rebuilt_markdown = rebuild_markdown(extracted_nodes, extracted_children)

    st.write("Rebuilt Markdown from Neo4j:")
    st.code(rebuilt_markdown)

    if content.strip() == rebuilt_markdown.strip():
        st.success("The structure is preserved!")
    else:
        st.error("The structure has changed.")
