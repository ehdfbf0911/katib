FROM tensorflow/tensorflow:1.14.0

ADD . /var/tf_cnn
ENTRYPOINT ["python3", "/var/tf_cnn/cnn-example.py"]
