FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common gnupg2
RUN add-apt-repository ppa:jonathonf/python-3.6 && apt-get update
RUN apt-get install python3.6 libpython3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN echo "America/Tijuana" >> /etc/timezone

RUN echo "deb http://ftp2.cn.debian.org/debian sid main contrib non-free" >> /etc/apt/sources.list
RUN echo "deb-src http://ftp2.cn.debian.org/debian sid main contrib non-free" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7638D0442B90D010
RUN apt-get update
RUN apt install -y caffe-cpu
COPY create_data.py /tmp
WORKDIR /tmp
ENTRYPOINT ["python3.6", 'create_data.py']