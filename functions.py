import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from googletrans import Translator
import torch
from langchain import HuggingFacePipeline
import textwrap
from langchain import PromptTemplate,  LLMChain
from numba import cuda
from GPUtil import showUtilization as gpu_usage
import pynvml
import PyPDF2
import whisper

####---------------GPU Functions------------------

def gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU Memory: {info.total / 1024**2} MB")
    print(f"Free GPU Memory: {info.free / 1024**2} MB")
    print(f"Used GPU Memory: {info.used / 1024**2} MB")
    pynvml.nvmlShutdown()

def free_gpu_cache():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")

        print("Initial GPU Usage")
        gpu_usage()
        torch.cuda.empty_cache()

        # Select the first CUDA device and close it
        cuda.select_device(0)
        cuda.close()

        # Select the first CUDA device again
        cuda.select_device(0)

        print("GPU Usage after emptying the cache")
        gpu_usage()
    else:
        print("CUDA is not available")

language_codes = {
    'Tamil': 'ta',
    'Hindi': 'hi',
    'Telugu': 'te',
    'English': 'en',
    'Nepali':'ne',
    'Punjabi':'pa',
    'Marathi':'mr',
    'Malyalam':'ml',
    'Kannada':'kn',
    'Odia':'or'

}


english_words = set(words.words())
def english_split(text):
    words = word_tokenize(text)
    
    # Check if there are no words in the text
    if len(words) == 0:
        return False
    
    english_word_count = sum(1 for word in words if word.lower() in english_words)
    english_ratio = english_word_count / len(words)
    threshold = 0.8
    return english_ratio >= threshold

def translate_input(text):
    trans=Translator()
    res=trans.translate(text,dest="en").text
    return res
def translate_output(text,out_lang):
    transout=Translator()
    res=transout.translate(text,src="auto",dest=out_lang).text
    return res



#########################---------------------------LLAM2 BOT-------------------------------------------------------##################

def initialize_llm_chain():
    model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.float16,
                                                device_map="cuda:0",
                                                revision="main")
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    # Define your system and instruction prompts with added spaces
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
        You are an advanced govt policy and planning expert that excels at giving advice.
        Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Just say you don't know and you are sorry!"""
    
    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT, citation=None):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    
        if citation:
            prompt_template += f"\n\nCitation: {citation}"  # Insert citation here
    
        return prompt_template
    
    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=512,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    system_prompt = "You are an advanced financial and savings expert that excels at giving advice."
    instruction = "Convert the following input text from a stupid human to a well-reasoned and step-by-step throughout advice:\n\n {text}"
    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(template=template, input_variables=["text"])
    
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)
    
    return llm_chain

#----------------------PDF Extraction--------------------------
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def wrap_text(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def pdf_bot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "meta-llama/Llama-2-7b-chat-hf"
    auth_token='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'
    tokenizer=AutoTokenizer.from_pretrained(name, use_auth_token=auth_token)
    model=AutoModelForCausalLM.from_pretrained(name,  use_auth_token=auth_token, torch_dtype=torch.float16, 
                                               rope_scaling={"type": "dynamic", "factor": 2},device_map="cuda:0")
    model.to(device)
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
        You are an advanced pdf chat expert that excels at giving advice.
        Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Just say you don't know and you are sorry!"""
    
    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT, citation=None):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    
        if citation:
            prompt_template += f"\n\nCitation: {citation}"  # Insert citation here
    
        return prompt_template
    
    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        max_new_tokens=512,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.7, 'max_length': 256, 'top_k': 50})
    return llm
     



######----------------------------------TTS WIth trasnlation voice-------------------------------#################
def transcribe_and_translate(input_text, dest_language):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("medium")
    model.to(device)
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(input_text, **translate_options)
    transcript = result['text']
    trans = Translator()
    translated_text = trans.translate(transcript, dest=dest_language)
    translated_result = translated_text.text
    return translated_result




######-------------------------------------VideoText Replacement ---------------------------#########

def whisper_load(input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("medium")
    model.to(device)
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(input, **translate_options)
    transcript = result['text']
    return transcript


def replace_word(input,target,replace_word):
    tokens=word_tokenize(input)
    tokens = [token if token.lower() != target.lower() else replace_word for token in tokens]
    result_text = ' '.join(tokens)
    return result_text


