from time import sleep
from istream_decoder import IStreamDecoder

from istream_player.config.config import PlayerConfig

video_file = open("/home/akram/ucalgary/research/istream-player/dataset/av1/bbb_250ms_30s_i_crf30.mp4", "rb")


decoder = IStreamDecoder()
decoder.setup(PlayerConfig)

while True:
    chunk = video_file.read(1 << 16)
    decoder.decode(chunk)

    frame = decoder.read()

    if frame is not None:
        print("GOT FRAME!!!!")
        sleep(1)