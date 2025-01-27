FROM ubuntu:22.04
LABEL Version="0.1"
LABEL Description="This is custom Docker Image for running a dash server."

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -q update && \
	apt-get -q install -y --no-install-recommends build-essential git libpcre3-dev libssl-dev zlib1g-dev gcc make ca-certificates software-properties-common

### Build FFMPEG FROM SCRATCH with AOM-AV1 and SVT-AV1 - Kyle
RUN apt-get -q install -y autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libgnutls28-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev meson ninja-build pkg-config texinfo wget yasm zlib1g-dev && \
        apt-get -q install -y nasm libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev libunistring-dev
RUN mkdir ~/ffmpeg_sources && \
        cd ~/ffmpeg_sources
# Add AOM
RUN git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom && \
        mkdir -p aom_build && \
        cd aom_build && \
        PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_SHARED=off -DENABLE_NASM=on ../aom && \
        PATH="$HOME/bin:$PATH" make -j 8 && \
        make install
# Add SVT-AV1
RUN cd ~/ffmpeg_sources && \
        git -C SVT-AV1 pull 2> /dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
        mkdir -p SVT-AV1/build && \
        cd SVT-AV1/build && \
        PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
        PATH="$HOME/bin:$PATH" make -j 8 && \
        make install
# Build ffmpeg with correct features
RUN cd ~/ffmpeg_sources && \
        wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
        tar xjvf ffmpeg-snapshot.tar.bz2 && \
        cd ffmpeg && \
        PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
          --prefix="$HOME/ffmpeg_build" \
          --pkg-config-flags="--static" \
          --extra-cflags="-I$HOME/ffmpeg_build/include" \
          --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
          --extra-libs="-lpthread -lm" \
          --bindir="$HOME/bin" \
          --enable-gpl \
          --enable-gnutls \
          --enable-libaom \
          --enable-libass \
          --enable-libfdk-aac \
          --enable-libfreetype \
          --enable-libmp3lame \
          --enable-libopus \
          --enable-libsvtav1 \
          --enable-libvorbis \
          --enable-libvpx \
          --enable-libx264 \
          --enable-libx265 \
          --enable-nonfree && \
        PATH="$HOME/bin:$PATH" make -j8 && \
        make install
# Add build to docker path
ENV PATH="${PATH}:/root/ffmpeg_sources/ffmpeg"


## These are behind paywalls now. Building it manually above.
#RUN add-apt-repository ppa:savoury1/ffmpeg5 && \
#	add-apt-repository ppa:savoury1/ffmpeg4
#RUN apt-get update
# RUN apt full-upgrade
# RUN apt-get install -y ffmpeg

RUN mkdir -p /tmp/build
RUN cd /tmp/build \
	&& git clone https://github.com/nginx/nginx.git \
	&& git clone https://github.com/arut/nginx-rtmp-module.git \
	&& cd nginx && ./auto/configure --add-module=../nginx-rtmp-module \
	&& make -s && make -s install

RUN cd /usr/local/nginx/conf && \
	rm nginx.conf

COPY nginx.conf /usr/local/nginx/conf

CMD ["./usr/local/nginx/sbin/nginx", "-g", "daemon off;"]
