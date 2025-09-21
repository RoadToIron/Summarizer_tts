from transformers import pipeline
import gradio as gr
import tensorflow as tf

#load the gpt model
talker = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
answerer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_and_tts(text):
    # Summarize the input text
    summary = answerer(text, max_length=50, min_length=25, do_sample=False)[0][text]
    
    # Convert the summary to speech
    tts_output = talker(summary)
    
    return summary, tts_output['audio']

# the Gradio Interface
with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Enter text to summarize and convert to speech", lines=10)
    summary_output = gr.Textbox(label="Summary")
    audio_output = gr.Audio(label="Text-to-Speech Output", type="numpy")
    submit_button = gr.Button("Submit")

demo.launch()