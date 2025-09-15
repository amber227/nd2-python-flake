from PIL import Image, ImageSequence
from random import shuffle

def make_gif_loop(input_path, output_path, last_frame_repeats=1):
    # Open the original GIF
    with Image.open(input_path) as im:
        # Extract frames
        frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
        # durations = [frame.info.get('duration', 100) for frame in ImageSequence.Iterator(im)]
        durations = [100 for frame in ImageSequence.Iterator(im)]
        print(len(durations))
        print(durations)
        # Repeat the last frame
        if last_frame_repeats > 1:
            # durations[-1] = durations[-1] * last_frame_repeats
            for _ in range(last_frame_repeats // 4):
                print("what")
                for i in range(len(frames) - 4,
                               len(frames)):
                    durations.append(durations[i])
                    frames.append(frames[i].copy())
            # frames.append(frames[0].copy())
            # durations.append(100)
            # durations.extend(durations[-50:])
            # frames.extend(durations[-50:])

        for i in range(0, len(durations)):
            durations[i] = int(durations[i] * 1.3)

        print(len(durations))
        print(durations)
        print(type(frames[0]))

        frames = frames[12:-4]
        durations = durations[12:-4]

        # Save new GIF with loop option
        frames[0].save(
            output_path,
            'GIF',
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,  # Infinite loop
            optimize=True
        )
        print(f'Looped GIF saved as: {output_path}')

# Usage example:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make a looping GIF with optional last frame repeats.")
    parser.add_argument("input", help="Input GIF path")
    parser.add_argument("output", help="Output GIF path")
    parser.add_argument("-r", "--last_frame_repeats", type=int, default=1,
                        help="Number of times to repeat the last frame (default: 1)")

    args = parser.parse_args()

    make_gif_loop(args.input, args.output, args.last_frame_repeats)
