import streamlit as st
import re
import tempfile
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_relationship(parent, child):
    """Generate a meaningful relationship between parent and child using GPT-3.5-turbo."""
    prompt = f"Given a parent node '{parent}' and a child node '{child}' in a mind map, suggest a meaningful one-word relationship between them. Just return the relationship word, nothing else."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates concise, meaningful relationships between concepts in a mind map."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    
    relationship = response.choices[0].message.content.strip().lower()
    return relationship

def add_relationships(content):
    """Add meaningful relationships to each line of the Markdown content while preserving structure."""
    lines = content.split('\n')
    updated_lines = []
    parent_stack = []
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            # Determine the level of the current line
            level = len(line) - len(line.lstrip())
            
            # Check if the line already has a relationship
            match = re.search(r'\(([^)]+)\)$', stripped_line)
            if match:
                existing_relationship = match.group(1)
                content = stripped_line[:match.start()].strip()
            else:
                existing_relationship = None
                content = stripped_line
            
            # If it's a root node or we don't have a parent, don't add a relationship
            if not parent_stack or level <= parent_stack[-1][0]:
                parent_stack = [(level, content)]
            else:
                parent = parent_stack[-1][1]
                if not existing_relationship:
                    relationship = generate_relationship(parent, content)
                    line = f"{line.rstrip()} ({relationship})"
                parent_stack.append((level, content))
        
        updated_lines.append(line)
    
    return '\n'.join(updated_lines)

# Streamlit app
st.title("Markdown Mind Map with Meaningful Relationships")

uploaded_file = st.file_uploader("Choose a Markdown file", type="md")

if uploaded_file is not None:
    content = uploaded_file.read().decode()
    
    with st.spinner("Processing... This may take a while as we generate meaningful relationships."):
        updated_markdown = add_relationships(content)
    
    st.success("Processing complete!")
    
    st.subheader("Updated Markdown with Meaningful Relationships")
    st.code(updated_markdown, language='markdown')
    
    st.subheader("Download Updated Markdown")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as tmpfile:
        with open(tmpfile.name, 'w', encoding='utf-8') as f:
            f.write(updated_markdown)
        
        with open(tmpfile.name, 'rb') as f:
            st.download_button(
                label="Download Updated Markdown",
                data=f,
                file_name="updated_mind_map.md",
                mime="text/markdown"
            )

st.sidebar.info("Upload a Markdown file to generate a mind map with meaningful relationships.")
