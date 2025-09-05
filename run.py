import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import cv2

# Load pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your image path
image = Image.open("input.png").convert("RGB")

# Generate video frames
frames = pipe(image, num_frames=14, decode_chunk_size=8).frames[0]

# Save as MP4
h, w, _ = frames[0].shape
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
for f in frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
out.release()

print("✅ Video saved as output.mp4")
