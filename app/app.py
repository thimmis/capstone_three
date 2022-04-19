
import streamlit as st
import requests
from PIL import Image


st.set_page_config(
    page_title="Summarization App",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

text_desc ="""
    The origin of this project comes from first and second-hand experience with
     the U.S. health care system, not because the quality of care is low, but
      rather because the continuity of care and access to health records between
       providers and in emergency settings is limited. In certain situations, a 
       better understanding of our past procedures and medical history and how 
       these relate to our lived experience could mean the difference between 
       catching a fully ruptured patellar tendon and being sent home with a 
       sprained knee for two weeks. The idea is to instill individuals with a 
       sense of ownership over their health outcomes and to provide them with 
       the ability to ask informed questions about their care, know when to ask 
       about certain procedures, another round of imaging, or even when to ask 
       for a second opinion.
"""

placeholder_text = """
    Patient was identified, then taken into the operating room, where after induction of appropriate anesthesia, his abdomen was prepped with Betadine solution and draped in a sterile fashion. The wound opening where it was draining was explored using a curette. The extent of the wound marked with a marking pen and using the Bovie cautery, the abscess was opened and drained. I then noted that there was a significant amount of undermining. These margins were marked with a marking pen, excised with Bovie cautery; the curette was used to remove the necrotic fascia. The wound was irrigated; cultures sent prior to irrigation and after achievement of excellent hemostasis, the wound was packed with antibiotic-soaked gauze. A dressing was applied. The finished wound size was 9.0 x 5.3 x 5.2 cm in size. Patient tolerated the procedure well. Dressing was applied, and he was taken to recovery room in stable condition.
"""

##helper funcs

def _check_server_status(url_path):
    server_down = '<p style="font-family:sans-serif; color:Red; font-size: 14px;">Not Running</p>'
    server_up = '<p style="font-family:sans-serif; color:Green; font-size: 14px;">Running</p>'
    
    server_code = requests.get(url_path).status_code
    return server_down if server_code == 404 else server_up


#image = Image.open('./banner.JPG')

#st.image(image,use_column_width=True)

def main():

    base_url = 'https://d719f3c297d8a4edb89d7f67aebe14091.clg07azjl.paperspacegradient.com/'
    sum_url = base_url+'v1/models/t5-summarizer:summarize'
    server_status = base_url+'healthcheck'

    

    html_title = """
    <div style="background:#460252 ;padding:10px">
    <h2 style="color:white;text-align:center;">Document Summarizer App</h2>
    </div>
    """

    st.markdown(html_title,  unsafe_allow_html = True)
    st.subheader('Transfer Learning with Transformers on Medical Documents')
    st.write(text_desc)

    st.write("""
    The model is a T5-model for conditional generation fine-tuned on a
     small selection of medical transcriptions. The maximum length of text the 
     can summarize is 512 words including punctuation, and it will return a 
     summary that is at most 128 words long also including punctuation.
     """)
    
    st.subheader('Server Status:')
    
    st.markdown(_check_server_status(server_status), unsafe_allow_html=True)
    message = st.text_area(
        "Input Text",
         height=250,
         value=placeholder_text)

    if st.button('Summarize'):

        payload = {
            'signature_name': 'serving_default',
            'text_data': {
                'data': message
            }
        }
        
        received = requests.post(sum_url, json=payload)

        st.write(received.text)


if __name__ == "__main__":
    main()
