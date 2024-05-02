
import streamlit as st
from mycrew import generate_response
import base64
from pathlib import Path
import re
import os
import shutil

def delete_images():
    folder = 'images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def replace_img_markdown(markdown):
    pattern = r'(!\[.*?\]\((.*?)\))'
    matches = re.findall(pattern, markdown)
    for match in matches:
        html_img = img_to_html(match[1])
        markdown = markdown.replace(match[0], html_img)
    return markdown

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.title("🦜 CrewAI Stock Price Demo Agents")


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = []
    st.session_state["api_key"] = ""

if user_prompt := st.chat_input():
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        response = generate_response(user_prompt)
        st.markdown(replace_img_markdown(response), unsafe_allow_html=True)

        # delete images folder contents
        delete_images()