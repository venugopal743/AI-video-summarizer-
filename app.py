import os
import whisper
import cv2 
from yt_dlp import YoutubeDL
from transformers import BartTokenizer, BartForConditionalGeneration 
from pymongo import MongoClient
from datetime import datetime
import logging
import torch 
from googletrans import Translator
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Load Whisper model and BART model once at the start
whisper_model = whisper.load_model("base")
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_model.to(device)

# MongoDB connection setup
mongo_client = MongoClient("mongodb+srv://shyamala:wyj.hdNrX8hGDub@cluster0.0f6zw.mongodb.net/")
db = mongo_client["video_database"]
collection = db["transcripts"]

def generate_summary(text):
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = bart_model.generate(inputs, max_length=600, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def download_video_and_transcribe(url):
    """
    Downloads the video from the provided URL using yt-dlp and returns the transcript using Whisper.
    """
    try:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': 'downloads/%(id)s.%(ext)s',  # Save video to downloads folder
            'noplaylist': True,  # Disable playlist download
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_file = ydl.prepare_filename(info_dict)

        # Transcribe the downloaded video using Whisper
        result = whisper_model.transcribe(video_file)
        transcript = result['text']

        # Clean up the downloaded video file
        os.remove(video_file)

        return video_file, transcript
    except Exception as e:
        logging.error(f"Error downloading or transcribing video: {e}")
        return None, None

def get_summary_for_time_range(full_summary, start_time, end_time):
    """
    Placeholder function to simulate time-based extraction from full summary.
    Modify this according to how your system stores timestamps.
    """
    summary_length = len(full_summary)
    range_length = int(summary_length * (end_time - start_time) / 1000)  # Assume time is in seconds
    return full_summary[:range_length]
@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.json
    video_url = data.get('url')

    # Check if the URL exists in the database
    existing_record = collection.find_one({"url": video_url})
    if existing_record:
        return jsonify({'exists': True})  # URL exists in the database
    else:
        return jsonify({'exists': False})  # URL does not exist

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dynamic_summary')
def dynamic_summary():
    # This route will display dynamic.html, you can pass additional data if needed
    return render_template('dynamic.html')

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    video_url = data.get('url')

    existing_record = collection.find_one({"url": video_url})
    if existing_record:
        return jsonify({'summary': existing_record['summary']})

    video_id, transcript = download_video_and_transcribe(video_url)
    if transcript:
        summary = generate_summary(transcript)
        collection.insert_one({
            "url": video_url,
            "transcript": transcript,
            "summary": summary,
            "date_processed": datetime.utcnow()
        })
        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'No transcript was generated.'})
def generate_summary(text):
    """
    Generate a concise summary of the input text.
    Limits the summary length to make it more concise.
    """
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True).to(device)  # Adjusted max_length
    summary_ids = bart_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def get_summary_for_time_range(transcript, start_time, end_time):
    """
    Extract the relevant portion of the transcript for the given time range and generate a concise summary.
    """
    # Split the transcript into sentences
    transcript_sentences = transcript.split('.')  # A simple split by sentences
    total_sentences = len(transcript_sentences)
    
    # Calculate the time duration of the range (in seconds)
    time_duration = end_time - start_time
    
    # Calculate the minimum number of sentences to extract based on the time range
    min_sentences = max(1, time_duration // 10)  # Ensures at least 1 sentence for small time ranges
    
    # Estimate the number of sentences per second (assuming uniform sentence distribution)
    sentence_per_second = total_sentences / (end_time - start_time)
    
    # Calculate the start and end sentence indices based on time
    start_sentence_idx = int(sentence_per_second * start_time)
    end_sentence_idx = start_sentence_idx + min_sentences  # Ensure we extract at least `min_sentences`
    
    # Extract the relevant portion of the transcript
    relevant_transcript = ' '.join(transcript_sentences[start_sentence_idx:end_sentence_idx])
    
    # Generate a concise summary for the extracted portion
    dynamic_summary = generate_summary(relevant_transcript)
    
    return dynamic_summary

@app.route('/get_dynamic_summary', methods=['POST'])
def get_dynamic_summary():
    data = request.json
    video_url = data.get('url')
    start_time = int(data.get('startTime'))
    end_time = int(data.get('endTime'))

    # Check if the video URL exists in the database
    existing_record = collection.find_one({"url": video_url})
    
    if existing_record:
        # If the URL exists, get the full transcript and summary
        full_transcript = existing_record['transcript']
        full_summary = existing_record['summary']
    else:
        # If the URL does not exist, download, transcribe, and summarize
        video_id, full_transcript = download_video_and_transcribe(video_url)
        
        if full_transcript:
            full_summary = generate_summary(full_transcript)
            
            # Store the URL, transcript, and summary in the database
            collection.insert_one({
                "url": video_url,
                "transcript": full_transcript,
                "summary": full_summary,
                "date_processed": datetime.utcnow()
            })
        else:
            return jsonify({'error': 'No transcript was generated.'}), 400
    
    # Extract the dynamic summary for the specified time range
    dynamic_summary = get_summary_for_time_range(full_transcript, start_time, end_time)
    
    # Return the dynamic summary
    return jsonify({'summary': dynamic_summary})

@app.route('/translate', methods=['POST'])
def translate_summary():
    data = request.json
    summary = data.get('summary')
    target_language = data.get('target_language')

    if not summary:
        return jsonify({'error': 'No summary provided'}), 400

    if not target_language:
        return jsonify({'error': 'Target language not provided'}), 400

    try:
        translator = Translator()
        translated_summary = translator.translate(summary, dest=target_language).text
        return jsonify({'translated_summary': translated_summary})

    except Exception as e:
        logging.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/supported-languages', methods=['GET'])
def get_languages():
    languages = {
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "zh": "Chinese",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ar": "Arabic",
        "ru": "Russian"
    }
    return jsonify({'languages': languages})
@app.route('/regenerate_summary', methods=['POST'])
def regenerate_summary():
    data = request.json
    translated_summary = data.get('summary')
    original_length = data.get('original_length')

    if not translated_summary or not original_length:
        return jsonify({'error': 'Summary or original length not provided'}), 400

    try:
        # Re-generate the summary with the same length as the original summary
        regenerated_summary = generate_summary_with_fixed_length(translated_summary, original_length)
        return jsonify({'translated_summary': regenerated_summary})

    except Exception as e:
        logging.error(f"Error regenerating summary: {e}")
        return jsonify({'error': 'Regeneration failed'}), 500

def generate_summary_with_fixed_length(text, original_length):
    """
    Generate a summary with a fixed length (in terms of words) to match the original summary's length.
    """
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = bart_model.generate(inputs, max_length=original_length + 50, min_length=original_length - 20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@app.route('/static-demo')
def static_demo():
    return send_file('static/static.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
