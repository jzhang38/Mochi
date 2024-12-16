from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

from genmo.lib.utils import save_video
import os
with open("prompt.txt", "r") as f:
    prompts = [line.rstrip() for line in f]
    
pipeline = MochiMultiGPUPipeline(
    text_encoder_factory=T5ModelFactory(),
    world_size=4,
    dit_factory=DitModelFactory(
        model_path=f"weights/dit.safetensors", model_dtype="bf16"
    ),
    decoder_factory=DecoderModelFactory(
        model_path=f"weights/decoder.safetensors",
    ),
)
# read prompt line by line from prompt.txt


output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
for i, prompt in enumerate(prompts):
    video = pipeline(
        height=480,
        width=848,
        num_frames=163,
        num_inference_steps=8,
        sigma_schedule=linear_quadratic_schedule(8, 0.1, 6),
        cfg_schedule=[1.5] * 8,
        batch_cfg=False,
        prompt=prompt,
        negative_prompt="",
        seed=12345,
    )[0]
    save_video(video, f"{output_dir}/output_{i}.mp4")
