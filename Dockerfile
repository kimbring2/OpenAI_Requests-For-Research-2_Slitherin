FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip

RUN python -m pip install --upgrade pip

WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
COPY opencv_python-4.2.0.34-cp35-cp35m-manylinux1_x86_64.whl opencv_python-4.2.0.34-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip install opencv_python-4.2.0.34-cp35-cp35m-manylinux1_x86_64.whl

COPY tensorflow_gpu-1.13.1-cp35-cp35m-manylinux1_x86_64.whl tensorflow_gpu-1.13.1-cp35-cp35m-manylinux1_x86_64.whl
RUN pip install tensorflow_gpu-1.13.1-cp35-cp35m-manylinux1_x86_64.whl

COPY . .

RUN pip install --default-timeout=1000 -r requirements.txt