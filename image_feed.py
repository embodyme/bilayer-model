import numpy as np
import cv2
import urllib.request
import logging as log
from threading import Thread
from datetime import datetime
from time import sleep


class VideoFeed:
    def __init__(self, display=False, post_processing=[], thread=True):
        self.img = None
        self.dirty = True
        self.started = False
        self.should_stop = False
        self.use_thread = thread
        self.display = display
        self.post_processing = post_processing
        self.timestamp = None
        return 

    def start(self):
        if not self.started:
            if self.use_thread:
                self.thread = Thread(target=self._update)
                self.thread.start()
                if self.display:
                    self.display_thread = Thread(target=self._display)
                    self.display_thread.start()
            self.started = True
            self.should_stop = False
        else:
            log.warn('Webcam stream already started')

    def _update(self):
        while not self.should_stop:
            self.__update()
            sleep(0.02)

    def __update(self):
        img = self._get_image()
        if img is None:
            return
        self.dirty = True
        for post in self.post_processing:
            img = post(img)
        self.img = img
        self.timestamp = datetime.now()

    def _get_images(self):
        return NotImplementedError()

    def kill(self):
        self.should_stop = True
        if self.use_thread:
            self.thread.join()
        if self.display:
            self.display_thread.join()

    def read(self):
        assert self.started, 'Call start before reading image from feed'
        if not self.use_thread:
            self.__update()
        _dirty = self.dirty
        self.dirty = False
        return self.img, _dirty, self.timestamp
        
    def _display(self):
        cv2.namedWindow(self.__class__.__name__, cv2.WINDOW_NORMAL)
        while not self.should_stop:
            if self.dirty and self.img is not None:
                cv2.imshow(self.__class__.__name__, self.img)
                cv2.waitKey(1)

    def save(self, filename, ext='jpg'):
        """Save the current image
        
        Parameters
        ----------
        filename : [type]
            [description]
        ext : str, optional
            [description], by default 'jpg'
        """
        filename = str(filename) + '.' + ext
        cv2.imwrite(filename, self.img)

class CVFeed(VideoFeed):
    def __init__(self, src=0, size=(512, 512), display=False, post_processing=[]):
        """A video camera object that provides a stream of frames"""
        super().__init__(display, post_processing)
        self.stream = cv2.VideoCapture(src)
        self.height, self.width = size
        self.stream.set(3, self.width)
        self.stream.set(4, self.height)

    def _get_image(self):
        (grabbed, img) = self.stream.read()
        if not grabbed:
            return None
        else:
            return img
