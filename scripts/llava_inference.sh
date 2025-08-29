DEFAULT_IMAGE_PATH="/home/ps/few-shot-research/AdaCLIP/test_image/006.jpg"

IMAGE_PATH=${1:-$DEFAULT_IMAGE_PATH}

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found at $IMAGE_PATH"
    echo "Please provide a valid image path as an argument:"
    echo "  ./llava_inference.sh /path/to/your/image.jpg"
    exit 1
fi

# Open vocabulary classification (automatically detect object categories)
python /home/ps/few-shot-research/mcxh_img/SegClip/scripts/llava_inference.py \
  --image_path "$IMAGE_PATH" \
  --llava_model /home/ps/llava-v1.5-7b \
  --fastsam_model /home/ps/few-shot-research/mcxh_img/SegClip/models/fastsam/FastSAM-x.pt \
  --open_vocabulary \
  --max_objects 100

# Or use predefined categories (uncomment the lines below and comment the lines above)
# python /home/ps/few-shot-research/mcxh_img/SegClip/scripts/llava_inference.py \
#   --image_path /home/ps/few-shot-research/mcxh_img/SegClip/asserts/desk.jpg \
#   --classes computer glass_bottle \
#   --llava_model /home/ps/llava-v1.5-7b \
#   --fastsam_model /home/ps/few-shot-research/mcxh_img/SegClip/models/fastsam/FastSAM-x.pt \
#   --max_objects 30