import subprocess
import json

def get_video_metadata(video_path):
    # Command to extract metadata using ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,sample_aspect_ratio,display_aspect_ratio,color_space",
        "-show_format",
        "-print_format", "json",
        video_path
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = json.loads(result.stdout)

    # Parse required fields
    stream = metadata['streams'][0]
    fmt = metadata['format']

    # Extract fields
    codec = stream.get('codec_name', 'N/A')
    format_name = fmt.get('format_name', 'N/A')
    width = stream.get('width', 0)
    height = stream.get('height', 0)
    resolution = f"{width}x{height}"
    duration = float(fmt.get('duration', 0.0))
    fps = eval(stream.get('r_frame_rate', '0/1'))
    par = stream.get('sample_aspect_ratio', 'N/A')
    colorspace = stream.get('color_space', 'N/A')
    aspect_ratio = stream.get('display_aspect_ratio', 'N/A')

    # Print results
    print(f"Codec        : {codec}")
    print(f"Format       : {format_name}")
    print(f"Resolution   : {resolution}")
    print(f"Duration     : {duration:.2f} seconds")
    print(f"FPS          : {fps:.2f}")
    print(f"PAR          : {par}")
    print(f"Colorspace   : {colorspace}")
    print(f"Aspect Ratio : {aspect_ratio}")

# Example usage
get_video_metadata("Aakrosh.mov")
