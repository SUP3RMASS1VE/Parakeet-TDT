import gradio as gr
import torch
import pandas as pd
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
import tempfile
import numpy as np
import subprocess
import math

# Function to load the parakeet TDT model
def load_model():
    # Load the model from HuggingFace
    print("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    print("Model loaded successfully!")
    return asr_model

# Global model variable to avoid reloading
model = None

def get_audio_duration(file_path):
    """Get the duration of an audio file using ffprobe"""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except (subprocess.SubprocessError, ValueError):
        return None

def extract_audio_from_video(video_path, progress=None):
    """Extract audio from video file"""
    # Use a dummy progress function if None provided
    if progress is None:
        progress = lambda x, desc=None: None
    
    progress(0.1, desc="Extracting audio from video...")
    
    # Create a temporary file for the extracted audio
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_path = temp_audio.name
    temp_audio.close()
    
    # Extract audio using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        audio_path,
        '-y'  # Overwrite if exists
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress(0.2, desc="Audio extraction complete")
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def split_audio_file(file_path, chunk_duration=600, progress=None):
    """Split audio into chunks of specified duration (in seconds)"""
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp()
    
    # Get total duration
    duration = get_audio_duration(file_path)
    if not duration:
        return None, 0
    
    # Calculate number of chunks
    num_chunks = math.ceil(duration / chunk_duration)
    chunk_files = []
    
    for i in range(num_chunks):
        if progress is not None:
            progress(i/num_chunks * 0.2, desc=f"Splitting audio ({i+1}/{num_chunks})...")
            
        start_time = i * chunk_duration
        output_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        
        # Use ffmpeg to extract chunk
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_file,
            '-y'  # Overwrite if exists
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk_files.append(output_file)
        except subprocess.CalledProcessError:
            # If error occurs, skip this chunk
            continue
    
    return chunk_files, duration

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
        try:
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
        except Exception:
            # If we can't read it as an audio file, it might be a video
            audio_path = audio_file
    
    # Check if audio is long (>50 minutes) and needs splitting
    duration = get_audio_duration(audio_path)
    long_audio_threshold = 600  # 10 minutes in seconds (changed from 3000/50min)
    chunk_duration = 600  # 10 minutes in seconds
    
    if duration and duration > long_audio_threshold:
        return process_long_audio(audio_path, is_music, progress, chunk_duration)
    
    # Normal processing for shorter audio
    return process_audio_chunk(audio_path, is_music, progress, 0, 1)

def transcribe_video(video_file, is_music=False, progress=gr.Progress()):
    """Transcribe the audio track from a video file"""
    # Extract audio from video
    audio_path = extract_audio_from_video(video_file, progress)
    if not audio_path:
        return "Error extracting audio from video", [], None
    
    # Now process the extracted audio
    return transcribe_audio(audio_path, is_music, progress)

def process_long_audio(audio_path, is_music, progress, chunk_duration):
    """Process long audio by splitting it into chunks"""
    # Split the audio file
    progress(0.1, desc="Analyzing audio file...")
    chunk_files, total_duration = split_audio_file(audio_path, chunk_duration, progress)
    
    if not chunk_files:
        return "Error splitting audio file", [], None
    
    # Process each chunk
    all_segments = []
    full_text_parts = []
    csv_data = []
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_start_time = i * chunk_duration
        progress_start = 0.2 + (i / len(chunk_files)) * 0.8
        progress_end = 0.2 + ((i + 1) / len(chunk_files)) * 0.8
        
        progress(progress_start, desc=f"Processing chunk {i+1}/{len(chunk_files)}...")
        
        # Process this chunk
        chunk_text, chunk_segments, _ = process_audio_chunk(
            chunk_file, 
            is_music, 
            progress, 
            chunk_start_time,
            progress_end - progress_start
        )
        
        # Add to results
        full_text_parts.append(chunk_text)
        all_segments.extend(chunk_segments)
        
        # Add to CSV data
        for segment in chunk_segments:
            csv_data.append({
                "Start (s)": f"{segment['start']:.2f}",
                "End (s)": f"{segment['end']:.2f}",
                "Segment": segment['text']
            })
        
        # Clean up chunk file
        try:
            os.unlink(chunk_file)
        except:
            pass
    
    # Clean up temp directory
    try:
        os.rmdir(os.path.dirname(chunk_files[0]))
    except:
        pass
    
    # Combine results
    full_text = " ".join(full_text_parts)
    
    # Create DataFrame for CSV
    df = pd.DataFrame(csv_data)
    
    # Save CSV to a temporary file
    csv_path = "transcript.csv"
    df.to_csv(csv_path, index=False)
    
    progress(1.0, desc="Done!")
    
    return full_text, all_segments, csv_path

def process_audio_chunk(audio_path, is_music, progress, time_offset=0, progress_scale=1.0):
    """Process a single audio chunk"""
    progress(0.3 * progress_scale, desc="Transcribing audio...")
    
    # Transcribe with timestamps
    output = model.transcribe([audio_path], timestamps=True)
    
    # Extract segment-level timestamps
    segments = []
    csv_data = []
    
    # Convert timestamp info to desired format
    if hasattr(output[0], 'timestamp') and 'segment' in output[0].timestamp:
        for stamp in output[0].timestamp['segment']:
            segment_text = stamp['segment']
            start_time = stamp['start'] + time_offset
            end_time = stamp['end'] + time_offset
            
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
    if time_offset == 0:  # Only save for first chunk or single chunk
        df.to_csv(csv_path, index=False)
    
    # Full transcript
    full_text = output[0].text if hasattr(output[0], 'text') else ""
    
    # Clean up the temporary file if created
    if isinstance(audio_path, str) and os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(audio_path)
        except:
            pass
    
    return full_text, segments, csv_path if time_offset == 0 else None

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
    gr.Markdown("# Audio & Video Transcription with Timestamps")
    gr.Markdown("Upload an audio/video file or record audio to get a transcript with timestamps")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("Upload Audio File"):
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            
            with gr.Tab("Upload Video File"):
                video_input = gr.Video(label="Upload Video File")
            
            with gr.Tab("Microphone"):
                audio_record = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="Record Audio",
                    show_label=True
                )
            
            is_music = gr.Checkbox(label="Music mode (better for songs)", info="Enable for more accurate song timestamps")
            audio_btn = gr.Button("Transcribe Audio", variant="primary")
            video_btn = gr.Button("Transcribe Video", variant="primary")
            gr.Markdown("""
            ### Notes:
            - Audio or video files over 10 minutes will be automatically split into smaller chunks for processing
            - Video files will have their audio tracks extracted for transcription
            - Splitting may take a few moments before transcription begins
            """)
        
        with gr.Column():
            full_transcript = gr.Textbox(label="Full Transcript", lines=5)
            transcript_segments = gr.JSON(label="Segments Data", visible=False)
            transcript_html = gr.HTML(label="Transcript Segments (Click a row to play)")
            csv_output = gr.File(label="Download Transcript CSV")
            audio_playback = gr.Audio(label="Audio Playback", elem_id="audio_playback", interactive=False)
    
    # Handle transcription from file upload
    audio_btn.click(
        transcribe_audio, 
        inputs=[audio_input, is_music],
        outputs=[full_transcript, transcript_segments, csv_output]
    )
    
    # Handle transcription from video
    video_btn.click(
        transcribe_video,
        inputs=[video_input, is_music],
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
    
    # Handle video audio extraction for playback
    video_input.change(
        lambda x: extract_audio_from_video(x, None) if x else None,
        inputs=[video_input],
        outputs=[audio_playback],
        js=js_code
    )

# Launch the app
if __name__ == "__main__":
    app.launch() 
