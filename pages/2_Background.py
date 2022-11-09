import streamlit as st

st.set_page_config(page_title = 'Background', page_icon=':shell:', layout='wide')
st.title('Background')
st.write("""You are working as an intern for an abalone farming operation in Japan. For operational and environmental reasons, it is an important consideration to estimate the age of the abalones when they go to market.\n
Determining an abalone's age involves counting the number of rings in a cross-section of the shell through a microscope. Since this method is somewhat cumbersome and complex, you are interested in helping the farmers estimate the age of the abalone using its physical characteristics.""")