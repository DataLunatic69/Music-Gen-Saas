from typing import List
import uuid
import modal
import os

from pydantic import BaseModel

from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT

app = modal.App("music-generator-diffrhythm")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "espeak-ng", "ffmpeg")
    .run_commands([
        "git clone https://github.com/ASLP-lab/DiffRhythm.git /tmp/DiffRhythm",
        "cd /tmp/DiffRhythm && pip install -r requirements.txt",
    ])
    .pip_install("supabase", "requests", "diffusers==0.32.2", "einops", "groq")
    .env({
        "HF_HOME": "/.cache/huggingface",
        "PYTHONPATH": "/tmp/DiffRhythm:/tmp/DiffRhythm/infer",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_python_source("prompts")
)

hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-generation-secret")


# ---------------------------------------------------------------------------
# LRC helpers
# ---------------------------------------------------------------------------

def plain_lyrics_to_lrc(lyrics: str, total_duration_s: float = 95.0) -> str:
    lines = [l.strip() for l in lyrics.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    intro_gap = 5.0
    usable = total_duration_s - intro_gap
    gap = usable / max(len(lines), 1)
    lrc_lines = []
    for i, line in enumerate(lines):
        t = intro_gap + i * gap
        minutes = int(t // 60)
        seconds = t % 60
        lrc_lines.append(f"[{minutes:02d}:{seconds:05.2f}]{line}")
    return "\n".join(lrc_lines)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AudioGenerationBase(BaseModel):
    audio_duration: float = 95.0
    seed: int = 42
    steps: int = 32
    cfg_strength: float = 3.8
    chunked: bool = True
    instrumental: bool = False


class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str


class GenerateMusicResponse(BaseModel):
    audio_url: str
    cover_image_url: str
    categories: List[str]


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15,
    timeout=900,  # 15 min — diffusion (2 min) + CPU/GPU transfers + upload
)
class MusicGenServer:

    @modal.enter()
    def load_model(self):
        import torch
        from groq import Groq
        from diffusers import StableDiffusionXLPipeline

        os.chdir("/tmp/DiffRhythm")

        self.device = "cuda"
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])

        # CRITICAL: patch out torch.compile BEFORE importing prepare_model.
        # prepare_model calls torch.compile(cfm) internally, creating a persistent
        # CUDA graph that holds ~78 GB outside PyTorch's normal allocator.
        # This memory is invisible to memory_allocated() and impossible to free
        # with empty_cache() — causing VAE decode to OOM even though the CFM
        # parameters have been moved to CPU.
        torch.compile = lambda model, *args, **kwargs: model

        from infer_utils import prepare_model

        (
            self.cfm,
            self.tokenizer_dr,
            self.muq,
            self.vae,
        ) = prepare_model(max_frames=2048, device="cpu")

        # MuQ and VAE park on GPU between requests; CFM stays on CPU
        self.muq = self.muq.to(self.device)
        self.vae = self.vae.to(self.device)

        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface",
        )
        self.image_pipe.to("cpu")

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _ask_llm(self, prompt: str) -> str:
        resp = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    def generate_prompt(self, description: str) -> str:
        return self._ask_llm(PROMPT_GENERATOR_PROMPT.format(user_prompt=description))

    def generate_lyrics(self, description: str) -> str:
        return self._ask_llm(LYRICS_GENERATOR_PROMPT.format(description=description))

    def generate_categories(self, description: str) -> List[str]:
        prompt = (
            f"Based on the following music description, list 3-5 relevant genres "
            f"or categories as a comma-separated list. For example: Pop, Electronic, "
            f"Sad, 80s. Description: '{description}'"
        )
        raw = self._ask_llm(prompt)
        return [c.strip() for c in raw.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_and_upload(
        self,
        style_prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        steps: int,
        cfg_strength: float,
        seed: int,
        chunked: bool,
        description_for_categorization: str,
    ) -> GenerateMusicResponse:
        import torch
        import gc
        import torchaudio
        from einops import rearrange
        from supabase import create_client
        from infer_utils import (
            decode_audio,
            get_lrc_token,
            get_negative_style_prompt,
            get_reference_latent,
            get_style_prompt,
        )

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)

        def _gpu_mb():
            return torch.cuda.memory_allocated() / 1024**2

        def _flush():
            """Force PyTorch to actually release CUDA memory."""
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # --- Build LRC ---
        if instrumental:
            lrc_content = ""
        else:
            lrc_content = plain_lyrics_to_lrc(lyrics, total_duration_s=audio_duration)

        print(f"Style prompt : {style_prompt}")
        print(f"LRC content  :\n{lrc_content}")

        max_frames = 2048
        audio_duration = min(audio_duration, 95.0)
        sample_rate = 44100

        # --- Tokenize + style embed (MuQ is on GPU) ---
        lrc_prompt, lrc_start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lrc_content, self.tokenizer_dr, audio_duration, self.device
        )
        style_prompt_token = get_style_prompt(self.muq, prompt=style_prompt)
        negative_style_prompt = get_negative_style_prompt(self.device)

        ref_latent, pred_frames = get_reference_latent(
            self.device, max_frames, False, None, None, None
        )

        # Move MuQ + VAE off GPU to make room for CFM
        self.muq.to("cpu")
        self.vae.to("cpu")
        _flush()
        print(f"[mem] after MuQ+VAE→CPU: {_gpu_mb():.0f} MB")

        # --- Move CFM to GPU for diffusion ---
        self.cfm.to(self.device)
        print(f"[mem] after CFM→GPU: {_gpu_mb():.0f} MB")

        from model.cfm import custom_mask_from_start_end_indices

        cfm = self.cfm
        cfm.eval()

        if next(cfm.parameters()).dtype == torch.float16:
            ref_latent = ref_latent.half()

        if ref_latent.shape[1] > end_frame:
            ref_latent = ref_latent[:, :end_frame, :]

        pred_segments_t = torch.tensor(pred_frames).to(self.device)
        fixed_span_mask = custom_mask_from_start_end_indices(
            end_frame, pred_segments_t, device=self.device, max_seq_len=end_frame
        ).unsqueeze(-1)
        step_cond = torch.where(fixed_span_mask, torch.zeros_like(ref_latent), ref_latent)

        torch.manual_seed(seed)
        y = torch.randn(1, end_frame, cfm.num_channels,
                        device=self.device, dtype=step_cond.dtype)
        t_steps = torch.linspace(0, 1, steps, device=self.device, dtype=step_cond.dtype)

        with torch.inference_mode():
            for i in range(len(t_steps) - 1):
                t_val = t_steps[i]
                dt = t_steps[i + 1] - t_steps[i]
                pred = cfm.transformer(
                    x=y, cond=step_cond, text=lrc_prompt, time=t_val,
                    drop_audio_cond=False, drop_text=False, drop_prompt=False,
                    style_prompt=style_prompt_token, start_time=lrc_start_time,
                    duration=song_duration,
                )
                null_pred = cfm.transformer(
                    x=y, cond=step_cond, text=lrc_prompt, time=t_val,
                    drop_audio_cond=True, drop_text=True, drop_prompt=False,
                    style_prompt=negative_style_prompt, start_time=lrc_start_time,
                    duration=song_duration,
                )
                velocity = pred + (pred - null_pred) * cfg_strength
                del pred, null_pred
                y = y + dt * velocity
                del velocity

        sampled = torch.where(fixed_span_mask, y, ref_latent)
        latent_result = sampled.to(torch.float32).clone().cpu()

        # Free all diffusion tensors from GPU
        del sampled, y, fixed_span_mask, step_cond, pred_segments_t
        del ref_latent, lrc_prompt, style_prompt_token
        del negative_style_prompt, lrc_start_time, song_duration
        del cfm  # drop local alias
        _flush()

        # Nuclear CUDA memory release sequence:
        # Step 1: move CFM parameters to CPU and zero gradients
        self.cfm.to("cpu")
        for p in self.cfm.parameters():
            p.grad = None
        # Step 2: sync CUDA stream so all ops are complete
        torch.cuda.synchronize()
        # Step 3: double flush — empty_cache twice forces the allocator
        #          to return blocks to the driver
        _flush()
        _flush()
        print(f"[mem] after CFM→CPU + sync: {_gpu_mb():.0f} MB")

        # --- VAE decode on GPU ---
        # synchronize() above ensures all euler loop activations are freed.
        # GPU only has 8 MB allocated now, so VAE (604 MB) fits easily.
        print(f"[mem] before VAE→GPU: {_gpu_mb():.0f} MB")

        self.vae.to(self.device)
        _flush()
        print(f"[mem] after VAE→GPU: {_gpu_mb():.0f} MB")

        latent = latent_result.to(self.device).float().transpose(1, 2)  # [b, d, t]
        del latent_result
        _flush()

        audio = decode_audio(latent, self.vae, chunked=True, chunk_size=64, overlap=16)
        del latent
        audio = rearrange(audio, "b d n -> d (b n)")
        generated_song = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        del audio
        _flush()

        # --- Upload audio ---
        audio_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(output_dir, audio_filename)
        torchaudio.save(output_path, generated_song, sample_rate=sample_rate)

        supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        bucket = os.environ["SUPABASE_BUCKET"]

        with open(output_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                audio_filename, f.read(),
                {"content-type": "audio/wav", "upsert": "true"},
            )
        os.remove(output_path)
        audio_url = supabase.storage.from_(bucket).get_public_url(audio_filename)

        # --- Cover art via SDXL-Turbo ---
        # VAE is already on CPU; image_pipe needs GPU
        self.image_pipe.to(self.device)
        thumb_prompt = f"{style_prompt}, album cover art"
        img = self.image_pipe(
            prompt=thumb_prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
        ).images[0]

        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(output_dir, image_filename)
        img.save(image_path)

        with open(image_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                image_filename, f.read(),
                {"content-type": "image/png", "upsert": "true"},
            )
        os.remove(image_path)
        cover_image_url = supabase.storage.from_(bucket).get_public_url(image_filename)

        # Restore idle state: MuQ + VAE on GPU, CFM + image_pipe on CPU
        self.image_pipe.to("cpu")
        _flush()
        self.muq.to(self.device)
        self.vae.to(self.device)

        categories = self.generate_categories(description_for_categorization)

        return GenerateMusicResponse(
            audio_url=audio_url,
            cover_image_url=cover_image_url,
            categories=categories,
        )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @modal.method()
    def run(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponse:
        lyrics = "" if request.instrumental else self.generate_lyrics(request.described_lyrics)
        return self._generate_and_upload(
            style_prompt=request.prompt,
            lyrics=lyrics,
            description_for_categorization=request.prompt,
            **request.model_dump(exclude={"prompt", "described_lyrics"}),
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_from_description(
        self, request: GenerateFromDescriptionRequest
    ) -> GenerateMusicResponse:
        style_prompt = self.generate_prompt(request.full_described_song)
        lyrics = "" if request.instrumental else self.generate_lyrics(request.full_described_song)
        return self._generate_and_upload(
            style_prompt=style_prompt,
            lyrics=lyrics,
            description_for_categorization=request.full_described_song,
            **request.model_dump(exclude={"full_described_song"}),
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(
        self, request: GenerateWithCustomLyricsRequest
    ) -> GenerateMusicResponse:
        return self._generate_and_upload(
            style_prompt=request.prompt,
            lyrics=request.lyrics,
            description_for_categorization=request.prompt,
            **request.model_dump(exclude={"prompt", "lyrics"}),
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_with_described_lyrics(
        self, request: GenerateWithDescribedLyricsRequest
    ) -> GenerateMusicResponse:
        lyrics = "" if request.instrumental else self.generate_lyrics(request.described_lyrics)
        return self._generate_and_upload(
            style_prompt=request.prompt,
            lyrics=lyrics,
            description_for_categorization=request.prompt,
            **request.model_dump(exclude={"prompt", "described_lyrics"}),
        )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    server = MusicGenServer()
    request_data = GenerateWithDescribedLyricsRequest(
        prompt="rave, funk, 140BPM, disco",
        described_lyrics="lyrics about water bottles",
        cfg_strength=3.8,
        steps=32,
        audio_duration=95.0,
    )
    result = server.run.remote(request_data)
    print(f"Audio  : {result.audio_url}")
    print(f"Cover  : {result.cover_image_url}")
    print(f"Tags   : {result.categories}")