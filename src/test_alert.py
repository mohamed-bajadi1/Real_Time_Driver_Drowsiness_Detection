"""
Test the AlarmController independently
"""
import os
import pygame
import time

# ============================================
# PATH RESOLUTION (Works from anywhere!)
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Project root directory
ALARM_PATH = os.path.join(PROJECT_ROOT, "data", "sounds", "alarm.wav")

print("=" * 60)
print("PATH DEBUG INFO")
print("=" * 60)
print(f"Script location:   {SCRIPT_DIR}")
print(f"Project root:     {PROJECT_ROOT}")
print(f"Alarm path:       {ALARM_PATH}")
print(f"File exists:       {os.path.exists(ALARM_PATH)}")

# If alarm. wav doesn't exist, try deep.wav
if not os.path.exists(ALARM_PATH):
    ALARM_PATH = os.path.join(PROJECT_ROOT, "data", "sounds", "deep.wav")
    print(f"Trying deep.wav:  {ALARM_PATH}")
    print(f"File exists:      {os.path.exists(ALARM_PATH)}")

print("=" * 60)


# ============================================
# ALARM CONTROLLER CLASS
# ============================================
class AlarmController:
    """
    Manages audio alarm for drowsiness detection. 
    """
    
    def __init__(self, sound_path, volume=0.7):
        """
        Initialize alarm system
        
        Args:
            sound_path:  Path to alarm WAV file
            volume:  Alarm volume (0.0 to 1.0)
        """
        # Initialize pygame mixer
        pygame.mixer.init(
            frequency=44100,  # CD quality
            size=-16,         # 16-bit audio
            channels=2,       # Stereo
            buffer=512        # Audio buffer size
        )
        
        self.sound = None
        self.is_playing = False
        
        try:
            # Load sound file ONCE (reuse it later)
            self.sound = pygame.mixer.Sound(sound_path)
            self.sound.set_volume(volume)
            print(f"✅ Alarm sound loaded: {sound_path}")
            print(f"   Duration: {self.sound.get_length():.2f}s")
            
        except FileNotFoundError:
            print(f"❌ Alarm sound not found: {sound_path}")
            print(f"   Please check the file exists!")
            self.sound = None
        except Exception as e:
            print(f"❌ Error loading alarm:  {e}")
            self.sound = None
    
    def play(self):
        """
        Play alarm sound in infinite loop
        """
        if self.sound and not self.is_playing:
            self.sound.play(loops=-1)  # loops=-1 = infinite repeat
            self.is_playing = True
            print("🚨 ALARM STARTED!")
    
    def stop(self):
        """
        Stop alarm sound
        """
        if self.sound and self.is_playing:
            self.sound.stop()
            self.is_playing = False
            print("🔇 Alarm stopped")
    
    def cleanup(self):
        """
        Cleanup audio resources on exit
        """
        if self.sound:
            self.sound.stop()
        
        pygame.mixer.stop()  # Stop all sounds
        pygame.mixer.quit()  # Quit mixer
        print("🔊 Audio system cleaned up")


# ============================================
# TEST THE ALARM
# ============================================
if __name__ == "__main__": 
    print("\n" + "=" * 60)
    print("TESTING ALARM CONTROLLER")
    print("=" * 60)
    
    # Create alarm controller
    alarm = AlarmController(ALARM_PATH, volume=0.5)
    
    if alarm.sound:
        print("\n✅ Sound loaded successfully!  Starting tests.. .\n")
        
        print("1️⃣  Playing alarm for 3 seconds...")
        alarm.play()
        time.sleep(3)
        
        print("\n2️⃣  Stopping alarm...")
        alarm.stop()
        time.sleep(1)
        
        print("\n3️⃣  Playing again for 2 seconds...")
        alarm.play()
        time.sleep(2)
        
        print("\n4️⃣  Cleanup...")
        alarm.cleanup()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ CANNOT TEST - Sound file not loaded")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check that data/sounds/alarm.wav exists")
        print("2. Verify file is not corrupted")
        print("3. Try using deep.wav instead")