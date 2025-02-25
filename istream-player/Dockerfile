FROM python:3.12

WORKDIR /src

# Download and install docker client
# RUN apt-get update && \
#     apt-get -y install apt-transport-https \
#          ca-certificates \
#          curl \
#          gnupg2 \
#          software-properties-common net-tools nano

# RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ENV headless=true

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# RUN git submodule update
RUN pip install .


RUN [ ! -e /run-dir ] && mkdir /run-dir || echo "Directory exists"

RUN mkdir plot-dir


USER $USER

# ENTRYPOINT ["/tests/resources/play360.sh"]
