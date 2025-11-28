FROM alpine:3.22.2
RUN ["mkdir", "/src"]
WORKDIR "/src"
RUN ["apk", "add", \
  "bash",\
  "python3",\
  "py3-pip",\
  "gcc",\
  "python3-dev",\
  "musl-dev",\
  "linux-headers"\
]
RUN ["python3", "-m", "venv", "/venv"]
COPY ./requirements.txt /requirements.txt
RUN ["/venv/bin/pip", "install", "-r", "/requirements.txt"]

COPY ./01_download_data.sh /01_download_data.sh
RUN [ "bash", "/01_download_data.sh" ]

# CMD [ "sh", "-c", "ls /src"]
#CMD [ "/venv/bin/python3", "/src/main.py" ]
CMD [ "/venv/bin/marimo", "edit", "-p", "6894", "--host", "0.0.0.0"]
