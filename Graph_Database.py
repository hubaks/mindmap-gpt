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
        if indent_level >= len(stack):
            stack.append(None)  # Add a placeholder if stack is not long enough
        
        parent_id = stack[indent_level - 1] if indent_level > 0 else None
        node_id = len(nodes)
        stack[indent_level:] = [node_id]  # Update stack to current level
        nodes.append((node_content, parent_id))
        
    return nodes

# Insert nodes into Neo4j
def insert_mind_map_into_neo4j(nodes):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
    with driver.session() as session:
        for node_id, (content, parent_id) in enumerate(nodes):
            session.write_transaction(create_node, node_id, content)
            if parent_id is not None:
                session.write_transaction(create_relationship, parent_id, node_id)
    driver.close()

# Create a node in Neo4j
def create_node(tx, node_id, content):
    tx.run("CREATE (n:Node {id: $id, content: $content})", id=node_id, content=content)

# Create a relationship in Neo4j
def create_relationship(tx, parent_id, child_id):
    tx.run("MATCH (a:Node {id: $parent_id}), (b:Node {id: $child_id}) "
           "CREATE (a)-[:HAS_CHILD]->(b)", parent_id=parent_id, child_id=child_id)

# Extract the mind map from Neo4j and rebuild the Markdown
def extract_mind_map_from_neo4j():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
    with driver.session() as session:
        result = session.run("MATCH (n:Node) OPTIONAL MATCH (n)-[:HAS_CHILD]->(m) RETURN n, collect(m) AS children")
        nodes = {}
        for record in result:
            node = record["n"]
            children = record["children"]
            # Handle potential None values in the children list
            if node is not None:
                children_ids = []
                for child in children:
                    if child is not None:
                        children_ids.append(child["id"])
                nodes[node["id"]] = (node["content"], children_ids)
    driver.close()
    return nodes

# Rebuild the Markdown structure
def rebuild_markdown(nodes):
    def build_markdown(node_id, indent=0):
        content, children = nodes[node_id]
        md_lines = [" " * indent + content]
        for child_id in sorted(children):
            md_lines.extend(build_markdown(child_id, indent + 2))
        return md_lines

    # Find all nodes without a parent
    root_nodes = [node_id for node_id, (content, children) in nodes.items() if not any(node_id in child for _, children in nodes.values())]
    markdown_lines = []
    for root_node in root_nodes:
        markdown_lines.extend(build_markdown(root_node))
    
    return "\n".join(markdown_lines)

# Streamlit app
st.title("Mind Map Graph Database POC")

uploaded_file = st.file_uploader("Upload a Markdown file", type="md")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    nodes = parse_mind_map(content)
    insert_mind_map_into_neo4j(nodes)

    st.write("Original Markdown:")
    st.code(content)

    extracted_nodes = extract_mind_map_from_neo4j()
    rebuilt_markdown = rebuild_markdown(extracted_nodes)

    st.write("Rebuilt Markdown from Neo4j:")
    st.code(rebuilt_markdown)

    if content.strip() == rebuilt_markdown.strip():
        st.success("The structure is preserved!")
    else:
        st.error("The structure has changed.")