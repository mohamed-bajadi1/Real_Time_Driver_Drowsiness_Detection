# ==================================
# IMPORTS (top of main_v4.py)
# ==================================
import pygame  # ✅ Import whole module, not pygame.mixer
import time

# ==================================
# CONFIGURATION
# ==================================
# Sound settings
ALARM_SOUND_PATH = "data\\sounds\\alarm.wav"  # ✅ No .. / needed if running from src/
ALARM_VOLUME = 0.7  # 70% volume
ENABLE_AUDIO = True  # Can disable for testing


# ==================================
# ALARM CONTROLLER CLASS
# ==================================
class AlarmController:
    """
    Manages audio alarm for drowsiness detection.
    
    Features:
    - Loads alarm sound once on initialization
    - Play/stop with state tracking
    - Prevents multiple simultaneous alarms
    """
    
    def __init__(self, sound_path, volume=0.7):
        """
        Initialize alarm system
        
        Args:
            sound_path:  Path to alarm WAV file
            volume: Alarm volume (0.0 to 1.0)
        """
        # Initialize pygame mixer
        pygame.mixer.init(
            frequency=44100,  # ✅ CD quality (NOT 441000!)
            size=-16,         # 16-bit audio
            channels=2,       # Stereo
            buffer=512        # Audio buffer size
        )
        
        self.sound = None
        self. is_playing = False
        
        try:
            # Load sound file ONCE (reuse it later)
            self.sound = pygame.mixer.Sound(sound_path)  # ✅ self.sound
            self.sound. set_volume(volume)                # ✅ self.sound
            print(f"✅ Alarm sound loaded: {sound_path}")
            print(f"   Duration: {self.sound.get_length():.2f}s")
            
        except FileNotFoundError: 
            print(f"❌ Alarm sound not found: {sound_path}")
            print(f"   Please check the file exists!")
            self.sound = None
        except Exception as e:
            print(f"❌ Error loading alarm:  {e}")
            self.sound = None
    
    def play(self):  # ✅ Don't forget self!
        """
        Play alarm sound in infinite loop
        
        Only plays if: 
        - Sound was loaded successfully
        - Not already playing (prevents overlap)
        """
        if self.sound and not self.is_playing:  # ✅ Use self.sound
            self.sound.play(loops=-1)  # ✅ loops=-1 = infinite repeat
            self.is_playing = True
            print("🚨 ALARM STARTED!")
    
    def stop(self):  # ✅ Don't forget self!
        """
        Stop alarm sound
        """
        if self.sound and self.is_playing:  # ✅ Use self. is_playing
            self.sound. stop()
            self.is_playing = False
            print("🔇 Alarm stopped")
    
    def cleanup(self):  # ✅ Don't forget self!
        """
        Cleanup audio resources on exit
        """
        if self.sound:
            self.sound.stop()
        
        pygame.mixer.stop()  # Stop all sounds
        pygame.mixer. quit()  # Quit mixer
        print("🔊 Audio system cleaned up")



if __name__ == "__main__":
    print("Testing AlarmController...")
    
    # Create alarm controller
    alarm = AlarmController("data/sounds/alarm.wav", volume=0.5)
    if alarm.sound:
        print("\n1. Playing alarm for 3 seconds...")
        alarm.play()
        time.sleep(3)
        
        print("\n2. Stopping alarm...")
        alarm.stop()
        time.sleep(1)
        
        print("\n3. Playing again for 2 seconds...")
        alarm.play()
        time.sleep(2)
        
        alarm.cleanup()
        print("\n✅ Test complete!")
    else:
        print("❌ Cannot test - sound file not loaded")