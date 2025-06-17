# !/usr/bin/env python3

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import logging
import shutil
import subprocess
import sys
import traceback
import ffmpeg
import numpy as np
from UpScaleUtil import create_parser, super_resolution_upscale, sr_logger
import tempfile
from PIL import Image
from shutil import which

def super_resolution_upscale_video(argv):
	parser = create_parser()

	parser.add_argument('--cache-disk', action='store_true',   help='Use Disk Cache During the Upscaling process rather then system Memory (Legacy)',
	                    default=False, dest='use_cache_disk')
	parser.add_argument('--codec', type=str, help='Allow to set a specific type of codec for the video',
	                    default='hevc', dest='codec')
	parser.add_argument('--frames', type=int, help='Allow to override the number of frames to extract',
	                    default=-1, dest='frames')	

	parsed_args = parser.parse_args(args=argv)

	#
	sr_logger.setLevel(logging.INFO)
	if parsed_args.debug:
		sr_logger.setLevel(logging.DEBUG)

	sr_logger.info("Parsed Arguments")

	# Handle input file path
	input_filepaths = parsed_args.input_files
	if os.path.isdir(input_filepaths):
		sr_logger.info("Directory Path: " + str(input_filepaths))
		all_files = os.listdir(input_filepaths)
		base_bath = input_filepaths
		input_filepaths: list = [os.path.join(
			base_bath, path) for path in all_files]
	else:  # Convert to list
		sr_logger.info("File Path: " + str(input_filepaths))
		input_filepaths: list = [input_filepaths]

	output_dir = parsed_args.save_path

	program_path: str = 'ffmpeg' # TODO: remove when using ffmpeg binding only.

	if which('ffmpeg') is None:
		raise Exception("ffmpeg not found")


	for input_file_path in input_filepaths:
		video_path = input_file_path
		source_video_full_path = str(os.path.abspath(video_path))
		output_final_video = os.path.join(output_dir, os.path.basename(input_file_path))

		# Create directory
		output_file_dir_path  = os.path.dirname(output_final_video)
		if not os.path.exists(output_file_dir_path):
			os.mkdir(output_file_dir_path)

		sr_logger.debug("Output File Path %s", output_final_video)

		# Attempt to Extract video meta information
		try:
			vid = ffmpeg.probe(filename=source_video_full_path)
			streams = vid['streams']

			video_streams = list(filter(lambda x: x['codec_type'] == 'video', streams))
			video_stream = video_streams[0]

			audio_streams = list(filter(lambda x: x['codec_type'] == 'audio', streams))
			audio_stream = audio_streams[0] if len(audio_streams) > 0 else None

			fps_rate = eval(video_stream['avg_frame_rate'])
			video_width = video_stream['width']
			video_height = video_stream['height']
			source_r_frame_rate = video_stream['r_frame_rate']
			#print(video_stream)
			frame_count = 0

			source_extract_fps_rate = eval(video_stream['r_frame_rate'])
		except Exception:
			sr_logger.error("Failed to Extract Video Meta from %s | Skipping File", source_video_full_path)
			continue

		if not parsed_args.use_cache_disk:

			# Extract Video Frames TODO: make async 
			input_frame_process, _ = (
				ffmpeg.input(source_video_full_path, r=source_r_frame_rate).output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=parsed_args.frames, start_number=0).run(capture_stdout=True))
			video_data = np.frombuffer(input_frame_process, np.uint8).reshape([-1, video_height, video_width, 3])
			#
			try:
				sr_logger.info("Starting %s", source_video_full_path)
				with tempfile.TemporaryDirectory() as tmpdirname:

					sr_logger.info('Created temporary directory %s', str(tmpdirname))

					# Upscale and save to tmp
					sr_logger.info("Preparing Upscaling Routine")
					upscale_commands = []
					upscale_commands.extend(['--input-file', "video_data"])
					upscale_commands.extend(['--save-output', "./"])
					upscale_commands.extend(['--model', parsed_args.model_filepath])

					# Convert the upscale.
					output_upscale_video_no_audio_path = os.path.join(tmpdirname, 'upscale_video.mp4')

					sr_logger.info(output_upscale_video_no_audio_path)
					output_process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{video_width * 2}x{video_height * 2}",  r=source_r_frame_rate ) \
					.output(output_upscale_video_no_audio_path, pix_fmt= 'yuv420p', vcodec=parsed_args.codec, crf=10, preset='slower', r=source_r_frame_rate) \
					.overwrite_output() \
					.run_async(pipe_stdin=True)

					def upscale_callback(arguments):
						try:
							#
							frame_index, upscale_image, new_cropped_size, full_output_path = arguments
						
							# Crop image final size image.
							upscale_image : Image = upscale_image.crop(new_cropped_size)

							# Convert to correct data format.
							pixel_data  = np.array(upscale_image).reshape([video_width * 2, video_height * 2, 3]).astype(np.uint8)

							# 
							#buffer, err =
							output_process.stdin.write(pixel_data.tobytes())

						except Exception as excep:
							sr_logger.error(excep)

					super_resolution_upscale(upscale_commands, np=video_data, upscale_callback=upscale_callback)

					output_process.stdin.close()
					output_process.wait()

					# Extract Audio
					extract_audio_path = os.path.join(tmpdirname, "audio.wav")
					sr_logger.debug("Audio Extract Path: %s", extract_audio_path)
					if audio_stream is not None:
						try:
							ffmpeg.input(source_video_full_path).output(extract_audio_path, vn=True, acodec='copy', f='wav').overwrite_output().run(capture_stdout=True, capture_stderr=True)
						except Exception as err:
							print(err.stderr)

					if extract_audio_path: # TODO: add pipe support
						# Merge upscale and Audio.
						source_audio = ffmpeg.input(extract_audio_path)
						upscaled_video = ffmpeg.input(source_video_full_path)

						outputstream = ffmpeg.output(upscaled_video, source_audio, output_final_video, vcodec='copy', acodec='copy').overwrite_output()
						ffmpeg.run(outputstream, quiet=False)  


					else:
						sr_logger.info("No Audio Track, transfering file directly without any video/audio Merging")
						shutil.move(output_upscale_video_no_audio_path, output_final_video)

			except Exception as ex:
				sr_logger.error("Error occurred during up-scaling of %s", source_video_full_path)
				sr_logger.error(ex)
				traceback.print_exc()
		else: # Use Disk Cache
			frame_pattern = 'frame_%04d.png'
			try:
				sr_logger.info("Starting %s", source_video_full_path)
				with tempfile.TemporaryDirectory() as tmpdirname:

					sr_logger.info('Created temporary directory %s', str(tmpdirname))

					source_image_path = os.path.join(tmpdirname, 'source')
					upscale_image_path = os.path.join(tmpdirname, 'upscale')

					sr_logger.debug(source_image_path)
					os.mkdir(source_image_path)
					sr_logger.debug(upscale_image_path)
					os.mkdir(upscale_image_path)


					# Save images to tmp directory
					sr_logger.debug("Prepare to Extract Video Frames to %s", source_image_path)
					source_extract_destination_pattern = os.path.join(source_image_path, frame_pattern)
					video_image_extract_command: list = []
					source_extract_fps_rate = str.format("fps={0}", fps_rate)
					video_image_extract_command.extend(
						[program_path, '-i', source_video_full_path, '-vf', source_extract_fps_rate,
						source_extract_destination_pattern])
					result = subprocess.call(video_image_extract_command, stdout=subprocess.PIPE)



					# Check if extraction of files successed
					if result != 0:
						sr_logger.error("Failed to extract image frames from video %s | Skipping File", source_video_full_path)
						os.rmdir(source_image_path)
						continue

					# Convert the upscale.
					upscale_extract_destination_pattern = os.path.join(upscale_image_path, frame_pattern)
					output_upscale_video_no_audio_path = os.path.join(tmpdirname, 'upscale_video.mp4')

					# Upscale and save to tmp
					sr_logger.info("Preparing Upscaling Routine")
					upscale_commands = []
					upscale_commands.extend(['--input-file', upscale_extract_destination_pattern])
					upscale_commands.extend(['--save-output', upscale_image_path])
					upscale_commands.extend(['--model', parsed_args.model_filepath])

					super_resolution_upscale(upscale_commands)

					# TODO: fix frame rate, being out of sync.
					# '24/1.001'
					create_video_command = []
					create_video_command.extend(
						[program_path, '-start_number', '1', '-framerate', '24', '-i', upscale_extract_destination_pattern,
						 '-c:v', 'libx265', '-crf', '15', '-r', source_extract_fps_rate, '-pix_fmt', 'yuv420p',
						 output_upscale_video_no_audio_path])
					result = subprocess.call(create_video_command, stdout=subprocess.PIPE)
					if result != 0:
						sr_logger.error("Failed to construct video from the upscaled images | Skipping file:")
						continue
 
					is_audio_extracted = False
					if audio_stream is not None:
				 
						audio_filename = str.format("video_audio.{}", audio_stream['codec_name'])
						output_audio = os.path.join(tmpdirname, audio_filename)
						extract_audio_args = [program_path]
						extract_audio_args.extend(['-i', video_path, '-vn', '-acodec', 'copy', output_audio])


						result = subprocess.call(extract_audio_args, stdout=subprocess.PIPE)
						if result == 0:
							is_audio_extracted = True

					if is_audio_extracted:  # TODO: pipe
						# Merge upscale and Audio.
						merge_video_audio_args = [program_path]
						merge_video_audio_args.extend(
							['-i', output_upscale_video_no_audio_path, '-i', output_audio, '-map', '0:v', '-map', '1:a', '-c:v',
							'copy',
							'-shortest', '-y',
							output_final_video])
						result = subprocess.call(merge_video_audio_args, stdout=subprocess.PIPE)

						#
						if result != 0:
							sr_logger.error("Failed to merge video and audio together | Skipping file")
							continue
					else:
						sr_logger.info("No Audio Track, transfering file directly")
						shutil.move(output_upscale_video_no_audio_path, output_final_video)

					# CLEANUP Section
					# Remove tmp original and upscaled files.
					os.remove(output_audio)
					os.remove(output_upscale_video_no_audio_path)

			except Exception as ex:
				sr_logger.error("Error occurred during up-scaling of %s", source_video_full_path)
				sr_logger.error(ex)
				traceback.print_exc()



# If running the script as main executable
if __name__ == '__main__':
	try:
		super_resolution_upscale_video(sys.argv[1:])
	except Exception as ex:
		sr_logger.error(ex)
		print(ex)
		traceback.print_exc()




