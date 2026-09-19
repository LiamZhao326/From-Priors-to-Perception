# External GPU Editing Dependencies

The basic scene, prompt, API, and label scripts use `requirements.txt`.
`sam2_propainter_edit.py` additionally requires the following external projects.
Their repositories and model weights are intentionally not bundled.

## SAM 2

- Repository: `https://github.com/facebookresearch/sam2`
- Model used by the working pipeline: SAM 2.1 Hiera Large
- Checkpoint filename: `sam2.1_hiera_large.pt`
- Official minimums: Python 3.10, PyTorch 2.5.1, torchvision 0.20.1

Install PyTorch for the target CUDA version first, then install SAM 2 from its
checkout:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

## ProPainter

- Repository: `https://github.com/sczhou/ProPainter`
- Entry point used by the pipeline: `inference_propainter.py`

Follow the repository's installation instructions and download its released
weights into the locations expected by ProPainter. Pass the checkout path to
`sam2_propainter_edit.py` using `--propainter-dir`.

## System tools

`ffmpeg` and `ffprobe` must be available on `PATH`. CUDA, PyTorch, and
torchvision versions must be selected as a compatible set for the target GPU
host.
