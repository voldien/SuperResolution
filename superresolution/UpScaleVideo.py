# !/usr/bin/env python3
import logging
import os
import shutil
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

	# 
	sr_logger.setLevel(logging.INFO)
	if args.debug:
		sr_logger.setLevel(logging.DEBUG)
		
	sr_logger.info("Parsed Arguments")

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
		video_full_path = str(os.path.abspath(video_path))
		output_final_video = args.save_path

		#ffmpeg.input(input_file_path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)

		# Attempt to Extract video info
		try:
			vid = ffmpeg.probe(filename=video_full_path)
			streams = vid['streams']
			video_stream = list(filter(lambda x: x['codec_type'] == 'video', streams))[0]
			audio_streams = list(filter(lambda x: x['codec_type'] == 'audio', streams))
			audio_stream = audio_streams[0] if len(audio_streams) > 0 else None
			fps_rate = eval(video_stream['avg_frame_rate'])
		except Exception:
			sr_logger.info("Failed to Extract Video Meta from {} | Skipping File", video_full_path)
			continue
		
		# 
		try:
			sr_logger.info("Starting %s", video_full_path)
			with tempfile.TemporaryDirectory() as tmpdirname:
				sr_logger.info('Created temporary directory %s', str(tmpdirname))
				source_image_path = os.path.join(tmpdirname, 'source')
				upscale_image_path = os.path.join(tmpdirname, 'upscale')
				os.mkdir(source_image_path)
				os.mkdir(upscale_image_path)

				# Save images to tmp
				source_extract_destination_pattern = os.path.join(source_image_path, frame_pattern)
				video_image_extract_command: list = []
				source_extract_fps_rate = str.format("fps={0}", fps_rate)
				video_image_extract_command.extend(
					[program_path, '-i', video_full_path, '-vf', source_extract_fps_rate,
					 source_extract_destination_pattern])
				result = subprocess.call(video_image_extract_command, stdout=subprocess.PIPE)

				# 
				if result != 0:
					sr_logger.error("Failed to extract image frames from video %s | Skipping File", video_full_path)
					os.rmdir(source_image_path)
					continue

				# Upscale and save to tmp
				upscale_commands = []
				upscale_commands.extend(['--input-file', source_image_path])
				upscale_commands.extend(['--save-output', upscale_image_path])
				upscale_commands.extend(['--model', args.model_filepath])
				# TODO: update variables from parsed
				super_resolution_upscale(upscale_commands)

				# Convert the upscale.
				upscale_extract_destination_pattern = os.path.join(upscale_image_path, frame_pattern)
				output_upscale_video_path = os.path.join(tmpdirname, 'upscale_video.mp4')
				create_video_command = []
				source_extract_fps_rate = "30"

				#TODO: fix frame rate, being out of sync.
				# '24/1.001'
				create_video_command.extend(
					[program_path, '-start_number', '1', '-framerate', '24', '-i', upscale_extract_destination_pattern,
					 '-c:v', 'libx265', '-crf', '15', '-r', source_extract_fps_rate, '-pix_fmt', 'yuv420p',
					 output_upscale_video_path])
				result = subprocess.call(create_video_command, stdout=subprocess.PIPE)
				if result != 0:
					sr_logger.error("Failed to construct video from the upscaled images | Skipping file")
					continue

				# Extract Audio # TODO: pipe 
				extract_audio = False
				if audio_stream is not None:
					audio_filename = str.format("video_audio.{}", audio_stream['codec_name'])
					output_audio = os.path.join(tmpdirname, audio_filename)
					extract_audio_args = [program_path]
					extract_audio_args.extend(['-i', video_path, '-vn', '-acodec', 'copy', output_audio])
					result = subprocess.call(extract_audio_args, stdout=subprocess.PIPE)
					if result == 0:
						extract_audio = True

				if extract_audio: # TODO: pipe 
					# Merge upscale and Audio. 
					merge_video_audio_args = [program_path]
					merge_video_audio_args.extend(
						['-i', output_upscale_video_path, '-i', output_audio, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
						 '-shortest',
						 output_final_video])
					result = subprocess.call(merge_video_audio_args, stdout=subprocess.PIPE)

				# 
				if result != 0:
					sr_logger.error("Failed to merge video and audio together | Skipping file")
					continue
				else:
					shutil.move(output_upscale_video_path, output_final_video)
				
				# Remove tmp original and upscaled files.
				os.rmdir(source_image_path)
				os.remove(output_audio)
				os.rmdir(upscale_image_path)
		except Exception as ex:
			sr_logger.error("Error occurred during up-scaling of %s", video_full_path)
			sr_logger.error(ex)


# If running the script as main executable
if __name__ == '__main__':
	try:
		super_resolution_upscale_video(sys.argv[1:])
	except Exception as ex:
		sr_logger.error(ex)
		print(ex)
		traceback.print_exc()
