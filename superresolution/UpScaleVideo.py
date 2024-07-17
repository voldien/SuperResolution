# !/usr/bin/env python3
import os
import subprocess
import sys
import traceback
import ffmpeg
from UpScaleUtil import create_parser, super_resolution_upscale, sr_logger
import tempfile
from shutil import which


def super_resolution_upscale_video(argv):
	parser = create_parser()

	args = parser.parse_args(args=argv)

	# Handle input file path
	input_filepaths = args.input_files
	if os.path.isdir(input_filepaths):
		sr_logger.info("Directory Path: " + str(input_filepaths))
		all_files = os.listdir(input_filepaths)
		base_bath = input_filepaths
		input_filepaths: list = [os.path.join(
			base_bath, path) for path in all_files]
	else:  # Convert to list
		sr_logger.info("File Path: " + str(input_filepaths))
		input_filepaths: list = [input_filepaths]

	program_path: str = 'ffmpeg'

	if which('ffmpeg') is None:
		raise Exception("ffmpeg not found")
	frame_pattern = 'frame_%04d.png'

	for input_file_path in input_filepaths:
		video_path = input_file_path
		video_full_path = os.path.abspath(video_path)
		video_basename = os.path.basename(video_path)
		output_final_video = args.save_path

		with tempfile.TemporaryDirectory() as tmpdirname:
			print('created temporary directory', tmpdirname)
			source_image_path = os.path.join(tmpdirname, 'source')
			upscale_image_path = os.path.join(tmpdirname, 'upscale')
			os.mkdir(source_image_path)
			os.mkdir(upscale_image_path)

			# Extract video info
			vid = ffmpeg.probe(filename=video_full_path)
			streams = vid['streams']
			video_stream = list(filter(lambda x: x['codec_type'] == 'video', streams))[0]
			fps_rate = video_stream['avg_frame_rate']

			# Save images to tmp
			source_extract_destination_pattern = os.path.join(source_image_path, frame_pattern)
			video_image_extract_command: list = []
			source_extract_fps_rate = str.format("fps={0}", fps_rate)
			video_image_extract_command.extend(
				[program_path, '-i', video_full_path, '-vf', source_extract_fps_rate,
				 source_extract_destination_pattern])
			subprocess.call(video_image_extract_command, stdout=subprocess.PIPE)

			# Upscale and save to tmp
			upscale_commands = []
			upscale_commands.extend(['--input-file', source_image_path])
			upscale_commands.extend(['--save-output', upscale_image_path])
			upscale_commands.extend(['--model', args.model_filepath])
			# TODO: update variables from parsed
			super_resolution_upscale(upscale_commands)

			# Convert the upscale.
			upscale_extract_destination_pattern = os.path.join(upscale_image_path, frame_pattern)
			output_no_video = os.path.join(tmpdirname, 'video.mp4')
			create_video_command = []
			# '24/1.001'
			create_video_command.extend(
				[program_path, '-start_number', '1', '-framerate', fps_rate, '-i', upscale_extract_destination_pattern,
				 '-c:v', 'libx265', '-r', source_extract_fps_rate, '-vf', 'format=yuv420', output_no_video])
			# ffmpeg -start_number 1 -framerate 24 -i "output_upscale/%3d.png" -c:v libx265 -r 24/1.001 -vf format=yuv420p output0.mp4
			subprocess.run(create_video_command, stdout=subprocess.PIPE)

			# Extract Audio 
			output_audio = os.path.join(tmpdirname, video_basename)
			extract_audio_args = [program_path]
			extract_audio_args.extend(['-i', video_path, '-vn', '-acodec', 'copy', output_audio])

			subprocess.call(extract_audio_args, stdout=subprocess.PIPE)

			# Merge upscale and Audio.
			# ffmpeg -i video.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -shortest output.mp4
			merge_video_audio_args = [program_path]
			merge_video_audio_args.extend(
				['-i', output_no_video, '-i', output_audio, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-shortest',
				 output_final_video])
			subprocess.call(extract_audio_args, stdout=subprocess.PIPE)


# super_resolution_upscale()


# If running the script as main executable
if __name__ == '__main__':
	try:
		super_resolution_upscale_video(sys.argv[1:])
	except Exception as ex:
		sr_logger.error(ex)
		print(ex)
		traceback.print_exc()
