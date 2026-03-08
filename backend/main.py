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
    .pip_install("supabase", "requests", "diffusers", "einops")
    .env({
        "HF_HOME": "/.cache/huggingface",
        # Both the repo root and infer/ must be on the path so that
        # `from infer_utils import ...` and `from model import ...` resolve.
        "PYTHONPATH": "/tmp/DiffRhythm:/tmp/DiffRhythm/infer",
    })
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("diffrhythm-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-generation-secret")


# ---------------------------------------------------------------------------
# LRC (timestamped lyrics) helpers
# ---------------------------------------------------------------------------

def plain_lyrics_to_lrc(lyrics: str, total_duration_s: float = 95.0) -> str:
    """
    Convert plain lyrics (with section labels like [verse], [chorus]) into
    a basic LRC format that DiffRhythm expects, e.g.:
        [00:05.00]Line one
        [00:08.50]Line two
    We spread the lines evenly across the song duration, leaving a small
    intro gap at the start.
    """
    lines = [l.strip() for l in lyrics.strip().splitlines() if l.strip()]
    if not lines:
        return ""

    intro_gap = 5.0  # seconds before first lyric
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
# Request / Response models  (mirrors your original ACE-Step interface)
# ---------------------------------------------------------------------------

class AudioGenerationBase(BaseModel):
    audio_duration: float = 95.0          # DiffRhythm-base max is 95s, full model 285s
    seed: int = 42
    steps: int = 32                        # diffusion steps (32 is a good default)
    cfg_strength: float = 3.8             # classifier-free guidance scale
    chunked: bool = True                   # use chunked VAE decode (saves VRAM)
    instrumental: bool = False


class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str          # style/genre description used as text prompt for DiffRhythm
    lyrics: str          # plain lyrics (we convert to LRC internally)


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str


class GenerateMusicResponse(BaseModel):
    audio_url: str           # Supabase Storage public URL
    cover_image_url: str     # Supabase Storage public URL
    categories: List[str]


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="L40S",                           # 8 GB+ required; L40S (48 GB) is plenty
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15,
)
class MusicGenServer:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image

        # All DiffRhythm code uses relative paths — anchor to the repo root.
        os.chdir("/tmp/DiffRhythm")

        from infer_utils import prepare_model

        self.device = "cuda"

        # DiffRhythm full model — max_frames=6144 supports up to 285 s output
        (
            self.cfm,
            self.tokenizer_dr,
            self.muq,
            self.vae,
        ) = prepare_model(max_frames=6144, device=self.device)

        # Compile for faster repeated inference
        self.cfm = torch.compile(self.cfm)

        # Qwen2-7B-Instruct — lyrics + prompt generation
        llm_id = "Qwen/Qwen2-7B-Instruct"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface",
        )

        # SDXL-Turbo — cover thumbnail generation
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface",
        )
        self.image_pipe.to(self.device)

    # ------------------------------------------------------------------
    # LLM helpers (identical to original)
    # ------------------------------------------------------------------

    def _prompt_qwen(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.llm_tokenizer([text], return_tensors="pt").to(
            self.llm_model.device
        )
        generated = self.llm_model.generate(
            inputs.input_ids, max_new_tokens=512
        )
        generated = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated)
        ]
        return self.llm_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_prompt(self, description: str) -> str:
        return self._prompt_qwen(
            PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        )

    def generate_lyrics(self, description: str) -> str:
        return self._prompt_qwen(
            LYRICS_GENERATOR_PROMPT.format(description=description)
        )

    def generate_categories(self, description: str) -> List[str]:
        prompt = (
            f"Based on the following music description, list 3-5 relevant genres "
            f"or categories as a comma-separated list. For example: Pop, Electronic, "
            f"Sad, 80s. Description: '{description}'"
        )
        raw = self._prompt_qwen(prompt)
        return [c.strip() for c in raw.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # Core DiffRhythm generation + S3 upload
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

        # Build LRC or leave empty for instrumental
        if instrumental:
            lrc_content = ""
        else:
            lrc_content = plain_lyrics_to_lrc(lyrics, total_duration_s=audio_duration)

        print(f"Style prompt : {style_prompt}")
        print(f"LRC content  :\n{lrc_content}")

        # The loaded model was initialised at max_frames=6144 (up to 285 s).
        # audio_duration controls the actual output length via get_lrc_token.
        max_frames = 6144
        sample_rate = 44100

        # --- Tokenise inputs (fixed signatures) ---
        lrc_prompt, lrc_start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lrc_content, self.tokenizer_dr, audio_duration, self.device
        )
        # get_style_prompt requires the muq model as the first argument
        style_prompt_token = get_style_prompt(self.muq, prompt=style_prompt)
        negative_style_prompt = get_negative_style_prompt(self.device)

        # Unconditional reference latent (no audio reference / no edit mode)
        ref_latent, pred_frames = get_reference_latent(
            self.device, max_frames, False, None, None, None
        )

        # --- Run DiffRhythm diffusion (call cfm.sample directly so that
        #     user-supplied steps / cfg_strength are actually honoured) ---
        torch.manual_seed(seed)
        with torch.inference_mode():
            latents, _ = self.cfm.sample(
                cond=ref_latent,
                text=lrc_prompt,
                duration=end_frame,
                style_prompt=style_prompt_token,
                max_duration=end_frame,
                song_duration=song_duration,
                negative_style_prompt=negative_style_prompt,
                steps=steps,
                cfg_strength=cfg_strength,
                start_time=lrc_start_time,
                latent_pred_segments=pred_frames,
                batch_infer_num=1,
            )

        # Decode first (and only) latent → stereo int16 waveform
        latent = latents[0].to(torch.float32).transpose(1, 2)  # [b, d, t]
        audio = decode_audio(latent, self.vae, chunked=chunked)
        audio = rearrange(audio, "b d n -> d (b n)")
        generated_song = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        # Save WAV
        audio_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(output_dir, audio_filename)
        torchaudio.save(output_path, generated_song, sample_rate=sample_rate)

        # --- Upload to Supabase Storage ---
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

        # Thumbnail via SDXL-Turbo
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

        categories = self.generate_categories(description_for_categorization)

        return GenerateMusicResponse(
            audio_url=audio_url,
            cover_image_url=cover_image_url,
            categories=categories,
        )

    # ------------------------------------------------------------------
    # Endpoints  (same surface as before)
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_description(
        self, request: GenerateFromDescriptionRequest
    ) -> GenerateMusicResponse:
        """Generate a full song from a free-form description (prompt + lyrics
        are both LLM-generated)."""
        style_prompt = self.generate_prompt(request.full_described_song)
        lyrics = "" if request.instrumental else self.generate_lyrics(
            request.full_described_song
        )
        return self._generate_and_upload(
            style_prompt=style_prompt,
            lyrics=lyrics,
            description_for_categorization=request.full_described_song,
            **request.model_dump(exclude={"full_described_song"}),
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_lyrics(
        self, request: GenerateWithCustomLyricsRequest
    ) -> GenerateMusicResponse:
        """Generate with a user-supplied style prompt and plain lyrics."""
        return self._generate_and_upload(
            style_prompt=request.prompt,
            lyrics=request.lyrics,
            description_for_categorization=request.prompt,
            **request.model_dump(exclude={"prompt", "lyrics"}),
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_described_lyrics(
        self, request: GenerateWithDescribedLyricsRequest
    ) -> GenerateMusicResponse:
        """Generate with a user-supplied style prompt; lyrics are LLM-generated
        from a description."""
        lyrics = "" if request.instrumental else self.generate_lyrics(
            request.described_lyrics
        )
        return self._generate_and_upload(
            style_prompt=request.prompt,
            lyrics=lyrics,
            description_for_categorization=request.prompt,
            **request.model_dump(exclude={"prompt", "described_lyrics"}),
        )


# ---------------------------------------------------------------------------
# Local entrypoint for quick smoke-test
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    # Call the Modal method directly (.remote()) instead of going through the
    # HTTP endpoint — the HTTP endpoints require proxy auth (requires_proxy_auth=True)
    # and are intended for external callers (e.g. the frontend).
    server = MusicGenServer()

    request_data = GenerateWithDescribedLyricsRequest(
        prompt="rave, funk, 140BPM, disco",
        described_lyrics="lyrics about water bottles",
        cfg_strength=3.8,
        steps=32,
        audio_duration=95.0,
    )

    result = server.generate_with_described_lyrics.remote(request_data)

    print(f"Audio  : {result.audio_url}")
    print(f"Cover  : {result.cover_image_url}")
    print(f"Tags   : {result.categories}")