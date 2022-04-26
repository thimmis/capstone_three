# Transfer Learning to Summarize Medical Documents

## Description
For my final capstone project with Springboard I explored the benefits of transfer learning for NLP applications, in particular I chose to explore fine-tuning a T5 model to summarize medical documents. The final result of this project is a web application where users can interact with the model by uploading a medical document and generating a summary of the document.

The idea for this project comes from first and second-hand experience with different aspects medical systems around the world. In all of them it is important to advocate for your own care in certain situations, as doing so could spell the difference between pressing the doctors for an MRI to catch a ruptured patelar tendon and being sent home with a sprained knee. The hope is that by generating summaries of medical documents it highlights key aspects of tests and procedures and creates opportunity to learn as well as the instills the mindset to ask those crucial questions, or a sense of ownership over their care.

The app can be accessed [here](https://share.streamlit.io/thimmis/capstone_three/app/app.py).

## Data

For the first iteration of this project a small set of medical transcriptions from MTSamples.com [[1]](1) was used to fine-tune the model. Each transcription was preprocessed in a similar fashion to the one used by [Chen, Gong and Zhuk](2) such that the Findings and Indications were the input and Impression served as the output. A heuristic approach was used to extract similar data when any one of the three sections was missing.

## Model

Using the HuggingFace Transformers library a T5-model with the 't5-small' checkpoint was fine-tuned for 50 epochs. Model performance was assessed through the rouge-L f1 score as well as through visual inspection. The baseline model produced an average f1 score of 16.38 while the fine-tuned model produed an average f1 score of 18.64.

Visual inspection of the resulting summaries indicates that where both models performed well the fine-tuned model did better at distilling information or capturing other relevant pieces of information.


## References
<a id="1">[1]</a>
MTSamples.com. (2022). Medical Transcriptions samples and reports. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?select=mtsamples.csv

<a id="2">[2]</a> 
X. Chen, S. Gong, W. Zhuk (2021). 
Predicting Doctor’s Impression For Radiology
Reports with Abstractive Text Summarization 
Stanford NLP CS224 Final Project Reports for 2021. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report005.pdf