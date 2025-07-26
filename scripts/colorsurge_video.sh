CUDA_VISIBLE_DEVICES=0 \
python3 ./inference/video_colorization_pipeline_colorsurge.py \
    --input_video ./inputs/raw_segment_14_gray.mp4 \
    --model_path "./pretrain_models/colorsurge_tiny.pth" \
    --output "./results/color_video/raw_segment_14_color.mp4" \
    --model_type Tiny
    