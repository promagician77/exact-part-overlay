# Exact part overlay - technical draft

Static site (deploy this folder to Vercel, "Other" preset, no build command) plus the spike that produced every number on it.

## Re-run the spike
```
cd spike
pip install numpy pillow opencv-python onnx onnxruntime
python src/make_inputs.py      # test part photo, segmentation alpha, target image
python src/measure_vae.py      # Stable Diffusion autoencoder round trip (TAESD, official weights, no PyTorch)
python src/pipeline.py         # layers, exact composite, 7 checks -> out/results.json
```
`comfyui/part_overlay_api.json` is the ComfyUI workflow (API format, core nodes only).
Node 2 loads the mask; in production that becomes a SAM2 or BiRefNet node. The final
ImageCompositeMasked takes its pixels from the original photo, not from the sampler.

TAESD weights: github.com/madebyollin/taesd (MIT). The test part is generated, not a client photo.
