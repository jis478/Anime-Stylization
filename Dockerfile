FROM docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /app
ADD . /app

RUN apt-get update
RUN conda install -c conda-forge dlib
RUN pip install -r requirements.txt

# Expose port 
EXPOSE 6006

# Run the application:
CMD ["python3", "app.py"]
