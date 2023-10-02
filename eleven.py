from elevenlabs import clone, generate, play, set_api_key
from elevenlabs.api import History

set_api_key("8f96a58113b07003fcf761c98bfb2c3b")

voice = clone(
    name="Voice Name",
    description="A young and talented Stanford student",
    files=["./sample.mp3"],
)

play(audio)

history = History.from_api()
print(history)
