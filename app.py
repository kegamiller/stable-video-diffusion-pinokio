import os, tempfile, torch, gradio as gr
from PIL import Image
import imageio.v2 as imageio
from diffusers import StableVideoDiffusionPipeline

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        ).to(device)
    return pipe

def generate(img, num_frames, fps, decode_chunk_size):
    if img is None:
        raise gr.Error("Please upload an image.")
    p = get_pipe()
    image = img.convert("RGB")
    out = p(image, num_frames=int(num_frames), decode_chunk_size=int(decode_chunk_size))
    frames = out.frames[0]  # list of numpy arrays (RGB)
    tmpdir = tempfile.mkdtemp(prefix="svd_")
    out_path = os.path.join(tmpdir, "output.mp4")
    imageio.mimsave(out_path, frames, fps=int(fps))
    return out_path

with gr.Blocks(title="Stable Video Diffusion (img→video)") as demo:
    gr.Markdown("## Stable Video Diffusion (img→video)\nUpload a still image, set frames & FPS, then generate an MP4.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            frames = gr.Slider(8, 25, value=14, step=1, label="Frames")
            fps = gr.Slider(4, 24, value=8, step=1, label="FPS")
            chunk = gr.Slider(1, 16, value=8, step=1, label="Decode chunk size")
            go = gr.Button("Generate")
        with gr.Column():
            vid = gr.Video(label="Result (MP4)")
    go.click(generate, [inp, frames, fps, chunk], vid)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
