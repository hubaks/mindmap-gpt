import streamlit as st
from neo4j import GraphDatabase

def parse_mind_map(md_content):
    nodes = []
    stack = []
    children = {}
    node_order = []
    for line in md_content.splitlines():
        indent_level = len(line) - len(line.lstrip())
        node_content = line.strip()
        while stack and stack[-1][1] >= indent_level:
            stack.pop()
        node_id = len(nodes)
        nodes.append((node_content, indent_level, node_id))
        node_order.append(node_id)
        if stack:
            parent_id = stack[-1][0]
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(node_id)
        children[node_id] = []
        stack.append((node_id, indent_level))
    return nodes, children, node_order

def insert_mind_map_into_neo4j(nodes, children):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
    with driver.session() as session:
        try:
            # Batch insert nodes
            session.write_transaction(lambda tx: [tx.run("CREATE (n:Node {id: $id, content: $content, indent: $indent})",
                                                         id=node_id, content=content, indent=indent_level)
                                                  for content, indent_level, node_id in nodes])
            # Batch insert relationships
            session.write_transaction(lambda tx: [tx.run("MATCH (parent:Node {id: $parent_id}), (child:Node {id: $child_id}) "
                                                        "CREATE (parent)-[:HAS_CHILD]->(child)",
                                                      parent_id=parent_id, child_id=child_id)
                                                  for parent_id, child_ids in children.items() for child_id in child_ids])
        except Exception as e:
            st.error(f"An error occurred while inserting data into Neo4j: {e}")
    driver.close()

def extract_mind_map_from_neo4j():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
    with driver.session() as session:
        try:
            result = session.run(
                "MATCH (n:Node) "
                "OPTIONAL MATCH (n)-[:HAS_CHILD]->(child) "
                "RETURN n.id AS id, n.content AS content, n.indent AS indent, "
                "collect(child.id) AS children ORDER BY n.id"
            )
            nodes = {}
            children = {}
            node_order = []
            for record in result:
                node_id = record["id"]
                nodes[node_id] = (record["content"], record["indent"])
                children[node_id] = record["children"]
                node_order.append(node_id)
        except Exception as e:
            st.error(f"An error occurred while extracting data from Neo4j: {e}")
    driver.close()
    return nodes, children, node_order

def rebuild_markdown(nodes, children, node_order):
    def build_markdown(node_id, indent_level):
        content, _ = nodes[node_id]
        md_lines = [" " * indent_level + content]
        for child_id in children[node_id]:
            md_lines.extend(build_markdown(child_id, indent_level + 2))
        return md_lines

    markdown_lines = []
    root_nodes = [node for node in node_order if all(node not in child_list for child_list in children.values())]
    for root_node in root_nodes:
        markdown_lines.extend(build_markdown(root_node, 0))
        markdown_lines.append("")  # Add an empty line after each main section
    return "\n".join(markdown_lines).strip()

# Streamlit app
st.title("Mind Map Graph Database POC")

uploaded_file = st.file_uploader("Upload a Markdown file", type="md")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    nodes, children, node_order = parse_mind_map(content)
    insert_mind_map_into_neo4j(nodes, children)
    
    st.write("Original Markdown:")
    st.code(content)
    
    extracted_nodes, extracted_children, extracted_order = extract_mind_map_from_neo4j()
    rebuilt_markdown = rebuild_markdown(extracted_nodes, extracted_children, extracted_order)
    
    st.write("Rebuilt Markdown from Neo4j:")
    st.code(rebuilt_markdown)
    
    if content.strip() == rebuilt_markdown.strip():
        st.success("The structure is preserved!")
    else:
        st.error("The structure has changed.")
    
    if st.button("Show Relationships in Neo4j"):
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
        with driver.session() as session:
            try:
                result = session.run(
                    "MATCH (n:Node)-[r:HAS_CHILD]->(m:Node) "
                    "RETURN n.content AS parent, m.content AS child "
                    "LIMIT 25"
                )
                relationships = [(record["parent"], record["child"]) for record in result]
            except Exception as e:
                st.error(f"An error occurred while fetching relationships: {e}")
        driver.close()
        st.write("Sample of Relationships in Neo4j:")
        for parent, child in relationships:
            st.write(f"'{parent}' -> '{child}'")

