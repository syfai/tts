from flask import Flask, request, jsonify, send_file
from tortoise.api import TextToSpeech
from tortoise.utils.audio import save_wav
import os

app = Flask(__name__)

# Initialize Tortoise TTS
tts = TextToSpeech()

@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate TTS audio
    voice_samples, conditioning_latents = tts.load_voice("random")
    generated_audio = tts.tts(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents)
    
    # Save the generated audio
    audio_path = 'generated_audio.wav'
    save_wav(generated_audio.squeeze(0), audio_path)
    
    # Return the audio file
    return send_file(audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
