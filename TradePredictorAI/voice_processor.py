import os
import speech_recognition as sr
from gtts import gTTS
import pygame
from datetime import datetime
import tempfile
import streamlit as st
from typing import Optional

class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        pygame.mixer.init()
        
    def listen_for_speech(self) -> Optional[str]:
        """
        Listen for speech input and convert to text
        """
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.info("Listening... Speak your query.")
                audio = self.recognizer.listen(source, timeout=5)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    st.warning("Could not understand the audio")
                    return None
                except sr.RequestError:
                    st.error("Could not request results from speech recognition service")
                    return None
        except Exception as e:
            st.error(f"Error capturing audio: {str(e)}")
            return None

    def text_to_speech(self, text: str) -> None:
        """
        Convert text to speech and play it
        """
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(fp.name)
                
                # Play the audio
                pygame.mixer.music.load(fp.name)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                # Clean up
                pygame.mixer.music.unload()
                os.unlink(fp.name)
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")

voice_processor = VoiceProcessor()
