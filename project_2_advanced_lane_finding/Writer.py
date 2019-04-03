from threading import Thread
import cv2

# continuously read frames
class Writer:

    def __init__(self, src=0, frame_width=0, frame_height=0):
        self.frame = cv2.VideoWriter('output1.avi',
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     10,
                                     (frame_width, frame_height))
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def save(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
