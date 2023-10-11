from flask import Flask,render_template,request,jsonify,send_file
from functions import *
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import json
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from ttsmms import TTS
import wavio
app=Flask(__name__)
app.config['ALLOWED_EXTENSIONS']={'pdf'}
UPLOAD_FOLDER = 'static/input'
AUDIO_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER



def allow_file(filename):
    print(f"Checking if file {filename} is allowed")
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

#################-----------------Gpu Memory Releaser----------------------------####################
free_gpu_cache()
###-------------------------------------------

@app.template_filter()
def jinja2_enumerate(iterable):
    return enumerate(iterable)


@app.route('/la',methods=['GET'])
def indx():
    return render_template('index.html')

@app.route('/askbot',methods=['GET','POST'])
def qabot():

    return render_template('index.html')
######################------------Lama2 Bot Initialization----------------------############
chats_history=[]
####2.-------------------Llam2 bOT----------------
@app.route('/llambot',methods=['GET','POST'])
def llambot(): 
   
    if request.method=="POST":
        user_input=request.form['user_input']
        print(f"Received user input: {user_input}") 
        language=request.form['language']
        print(f"Received user Langauge: {language}") 
        
        chats_history.append(("User",user_input))
        print(f"User Question: {chats_history}")
        if english_split(user_input):
           prompt=user_input
           print(f"User text {prompt}")
        else :
            translated_text=translate_input(user_input)
            print(f"Trasnalted Input :{translated_text}")
            prompt=translated_text
        llm_chain = initialize_llm_chain()
        response=llm_chain.run(prompt)
        modified_response = "\n\n".join(response.split("\n"))
        #print(response)
        translated_response = translate_output(modified_response,language)
        print(translated_response)
        chats_history.append(("Bot",translated_response))
        return render_template('chat.html',chats_history=chats_history)
    return render_template('chat.html')


tts_models = {
    'hi': 'voice/hin',
    'pa': 'voice/pan', 
    'ta':'voice/tam',
    'mr':'voice/mar',
    'kn':'voice/kan',
    'ory':'voice/ory',
    'ur':'voice/urd-script_arabic',
    'en':'voice/eng',
}


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the uploaded file to the specified folder
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            dest_language = request.form.get('dest_language', 'en')  # Default to English if not provided
            # Transcribe and translate text
            translated_text = transcribe_and_translate(video_path, dest_language)
            # Clean up the temporary video file
            os.remove(video_path)
            tts_model_path = tts_models.get(dest_language, 'voice/eng')  # Default to English if not found
            # Use the selected TTS model path to generate audio
            tts = TTS(tts_model_path)
            wav = tts.synthesis(translated_text)
            audio_file = 'static/output_audio.wav'
            wavio.write(audio_file, wav["x"], wav["sampling_rate"], sampwidth=2)
            return send_file(audio_file, mimetype="audio/wav")

        else:
            return jsonify({'error': 'Invalid file format. Please upload an MP4 file.'})
    return render_template('sound.html')
    
@app.route('/replace',methods=['GET','POST'])
def text_edit():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        # Check if the file has a name
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check if the file is allowed (e.g., video formats)
        allowed_extensions = {'mp4', 'avi', 'mkv'}
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            # Save the uploaded file to a temporary location
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Get the original word and replacement word from the form
            original_word = request.form['original_word']
            replacement_word = request.form['replacement_word']

            # Call your transcribe_and_translate function with the uploaded video
            destination_language = "en"  # Replace with the desired destination language
            translated_result = transcribe_and_translate(file_path, destination_language)

            # Replace the original word with the replacement word
            translated_result = translated_result.replace(original_word, replacement_word)

            return render_template('result.html', result=translated_result)

        else:
            return jsonify({"error": "Invalid file format"}), 400

    return render_template('index.html')

   
app.run(debug=True)
