import gradio as gr
import torch
import pandas as pd
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
import tempfile
import numpy as np

# Function to load the parakeet TDT model
def load_model():
    # Load the model from HuggingFace
    print("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    print("Model loaded successfully!")
    return asr_model

# Global model variable to avoid reloading
model = None

def transcribe_audio(audio_file, is_music=False, progress=gr.Progress()):
    global model
    
    # Load the model if not already loaded
    if model is None:
        progress(0.1, desc="Loading model...")
        model = load_model()
    
    # Save the temporary audio file if it's from Gradio
    if isinstance(audio_file, tuple):
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        sample_rate, audio_data = audio_file
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        import soundfile as sf
        sf.write(temp_audio_path, audio_data, sample_rate)
        audio_path = temp_audio_path
    else:
        # For files uploaded directly, we need to convert if stereo
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            
            # For music, apply some preprocessing to improve vocal separation
            if is_music:
                try:
                    # Normalize audio
                    audio_data = audio_data / np.max(np.abs(audio_data))
                    
                    # Apply a simple high-pass filter to emphasize vocals (at 200Hz)
                    from scipy import signal
                    b, a = signal.butter(4, 200/(sample_rate/2), 'highpass')
                    audio_data = signal.filtfilt(b, a, audio_data)
                    
                    # Slight compression to bring up quieter vocals
                    threshold = 0.1
                    ratio = 0.5
                    audio_data = np.where(
                        np.abs(audio_data) > threshold,
                        threshold + (np.abs(audio_data) - threshold) * ratio * np.sign(audio_data),
                        audio_data
                    )
                except ImportError:
                    # If scipy is not available, skip preprocessing
                    pass
            
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            sf.write(temp_audio_path, audio_data, sample_rate)
            audio_path = temp_audio_path
        else:
            audio_path = audio_file
    
    progress(0.3, desc="Transcribing audio...")
    
    # Transcribe with timestamps - Removed decoder_params as it's not supported
    output = model.transcribe([audio_path], timestamps=True)
    
    # Extract segment-level timestamps
    segments = []
    
    # For CSV output
    csv_data = []
    
    # Convert timestamp info to desired format
    if hasattr(output[0], 'timestamp') and 'segment' in output[0].timestamp:
        for stamp in output[0].timestamp['segment']:
            segment_text = stamp['segment']
            start_time = stamp['start']
            end_time = stamp['end']
            
            # For music, we can do some post-processing of the timestamps
            if is_music:
                # Add a small buffer to ensure segments don't cut off too early
                # This helps with stretched syllables often found in singing
                end_time += 0.3
                
                # Minimum segment duration for lyrics
                min_duration = 0.5
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration
            
            segments.append({
                "text": segment_text,
                "start": start_time,
                "end": end_time
            })
            
            # Add to CSV data
            csv_data.append({
                "Start (s)": f"{start_time:.2f}",
                "End (s)": f"{end_time:.2f}",
                "Segment": segment_text
            })
    
    # Create DataFrame for CSV
    df = pd.DataFrame(csv_data)
    
    # Save CSV to a temporary file
    csv_path = "transcript.csv"
    df.to_csv(csv_path, index=False)
    
    # Full transcript
    full_text = output[0].text if hasattr(output[0], 'text') else ""
    
    progress(1.0, desc="Done!")
    
    # Clean up the temporary file if created
    if isinstance(audio_file, tuple) and os.path.exists(temp_audio_path):
        os.unlink(temp_audio_path)
    
    return full_text, segments, csv_path

def create_transcript_table(segments):
    if not segments:
        return "No segments found"
    
    html = """
    <style>
    .transcript-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .transcript-table th, .transcript-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .transcript-table th {
        background-color: #f2f2f2;
    }
    .transcript-table tr:hover {
        background-color: #f5f5f5;
        cursor: pointer;
    }
    </style>
    <table class="transcript-table">
        <tr>
            <th>Start (s)</th>
            <th>End (s)</th>
            <th>Segment</th>
        </tr>
    """
    
    for segment in segments:
        html += f"""
        <tr onclick="document.dispatchEvent(new CustomEvent('play_segment', {{detail: {{start: {segment['start']}, end: {segment['end']}}}}}))">
            <td>{segment['start']:.2f}</td>
            <td>{segment['end']:.2f}</td>
            <td>{segment['text']}</td>
        </tr>
        """
    
    html += "</table>"
    return html

# Define custom JavaScript to handle segment playback
js_code = """
function(audio) {
    document.addEventListener('play_segment', function(e) {
        const audioEl = document.querySelector('audio');
        if (audioEl) {
            audioEl.currentTime = e.detail.start;
            audioEl.play();
            
            // Optional: Stop at segment end
            const stopAtEnd = function() {
                if (audioEl.currentTime >= e.detail.end) {
                    audioEl.pause();
                    audioEl.removeEventListener('timeupdate', stopAtEnd);
                }
            };
            audioEl.addEventListener('timeupdate', stopAtEnd);
        }
    });
    return audio;
}
"""

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as app:
    gr.Markdown("# Audio Transcription with Timestamps")
    gr.Markdown("Upload an audio file or record audio to get a transcript with timestamps")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("Upload Audio File"):
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            
            with gr.Tab("Microphone"):
                audio_record = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="Record Audio",
                    show_label=True
                )
            
            is_music = gr.Checkbox(label="Music mode (better for songs)", info="Enable for more accurate song timestamps")
            transcribe_btn = gr.Button("Transcribe Uploaded File", variant="primary")
            gr.Markdown("""
            ### Notes:
            - For long audio files (>5 minutes), transcription may require significant memory else oom will occur
            - If you encounter memory errors, try processing shorter clips
            """)
        
        with gr.Column():
            full_transcript = gr.Textbox(label="Full Transcript", lines=5)
            transcript_segments = gr.JSON(label="Segments Data", visible=False)
            transcript_html = gr.HTML(label="Transcript Segments (Click a row to play)")
            csv_output = gr.File(label="Download Transcript CSV")
            audio_playback = gr.Audio(label="Audio Playback", elem_id="audio_playback", interactive=False)
    
    # Handle transcription from file upload
    transcribe_btn.click(
        transcribe_audio, 
        inputs=[audio_input, is_music],
        outputs=[full_transcript, transcript_segments, csv_output]
    )
    
    # Handle transcription from microphone
    audio_record.stop_recording(
        transcribe_audio,
        inputs=[audio_record, is_music],
        outputs=[full_transcript, transcript_segments, csv_output]
    )
    
    # Update the HTML when segments change
    transcript_segments.change(
        create_transcript_table,
        inputs=[transcript_segments],
        outputs=[transcript_html]
    )
    
    # Apply custom JavaScript for audio playback
    audio_input.change(
        lambda x: x,
        inputs=[audio_input],
        outputs=[audio_playback],
        js=js_code
    )
    
    audio_record.stop_recording(
        lambda x: x,
        inputs=[audio_record],
        outputs=[audio_playback],
        js=js_code
    )

# Launch the app
if __name__ == "__main__":
    app.launch() 
