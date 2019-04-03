from project_2_advanced_lane_finding.Reader import Reader
from project_2_advanced_lane_finding.Processor import Processor
from project_2_advanced_lane_finding import advanced_lane_finding_2

import cv2, time
# @author Blake Denniston (bd21)

"""
processing a video is essentially the following steps:
1. read a frame
2. process the frame
3. save the frame to memory

by separating tasks 1 and 2-3 in separate threads, we get 10000x speedups
"""


def main():
    # source = "short_videos/short.mp4"
    source = "Test_Inputs/project_video.mp4"
    fast_display(source)

def fast_display(source):

    reader = Reader(source).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or reader.stopped:
            reader.stop()
            break
        start = time.time()

        print(time.time() - start)
        frame = reader.frame

        print(time.time() - start)
        frame = advanced_lane_finding_2.process_frame(frame)
        
        print(time.time() - start)
        cv2.imshow("Video", frame)

        print(time.time() - start)
        print("end")


def fast_display_2(source):

    reader = Reader(source).start()
    processor = Processor(reader.frame).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or reader.stopped:
            reader.stop()
            processor.stop()
            break

        frame = reader.frame

        start = time.time()

        print(time.time() - start)
        frame = advanced_lane_finding_2.process_frame(frame)
        print(time.time() - start)

        processor.frame = frame
        print(time.time() - start)
        print("end")


# todo - unfinished
def fast_save(source):

    reader = Reader(source).start()
    processor = Processor(reader.frame).start()

    while True:
        if reader.stopped or processor.stopped:
            reader.stop()
            processor.stop()
            break


        frame = reader.frame
        processor.frame = frame

if __name__ == "__main__":
    main()
