from flask import Flask, render_template, redirect, url_for, request
import os
import sys
import bz2
import torch
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
# from tensorflow.keras.utils import get_file
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import time
import cv2 

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def align_image(image_name, size, landmarks_model_path):
    s = time.time()
    print('align start')
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    face_landmarks = landmarks_detector.get_landmarks(image_name)
    aligned_image = image_align(image_name, face_landmarks, size)
    e = time.time() - s
    print('align time:', e)
    return aligned_image


def inference(model, img_tensor, model_name):
    img_tensor = img_tensor.unsqueeze(0).cuda()    
    out = model(img_tensor)
    out = out.detach().cpu()
    img = Image.fromarray(tensor2im(out[0]))
    img.save(f'static/{model_name}.png')
    
    
def pix2pixHD_inference(aligned_images, size, h, w):
    s = time.time()
    transform = transforms.Compose([transforms.Resize([256, int(256 * h/w)]), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
    img_tensor = transform(aligned_images)    
    for model, name in zip(model_list, model_name_list):
        inference(model, img_tensor, name)
        e = time.time() - s
        print('inference time:', e)



def pix2pixHD_inference_ag(model, name, aligned_images, size, h, w):
    transform = transforms.Compose([transforms.Resize([256, int(256 * h/w)]), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
    img_tensor = transform(aligned_images)    
    inference(model, img_tensor, name)
    
def pipeline(image_name, size, landmarks_model_url):
    aligned_image, h, w = align_image(image_name, SIZE, landmarks_model_path)    
    aligned_image.save(aligned_name)
    pix2pixHD_inference(aligned_image, SIZE, h, w)
    img = Image.open(f'./static/ch_tb_32.png')
    pix2pixHD_inference_ag(model_list[0], 'ag_ch_tb_32', img, SIZE, h, w)
    img = Image.open(f'./static/sk_tb_16.png')
    pix2pixHD_inference_ag(model_list[1], 'ag_sk_tb_16', img, SIZE, h, w)
    img = Image.open(f'./static/sk_tb_32.png')
    pix2pixHD_inference_ag(model_list[2], 'ag_sk_tb_32', img, SIZE, h, w)

        
    
@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA').convert('RGB')
        image.save(image_name)
        pipeline(image_name, SIZE, landmarks_model_path)
        return render_template('result.html')
    return render_template('index.html')


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    
    SIZE = 256
    image_name = f'static/temp.png'
    aligned_name = f'static/aligned.png'
    landmarks_model_path = 'models/shape_predictor_68_face_landmarks.dat'
    model1 = torch.jit.load('models/ch_tb.pt')   
    model2 = torch.jit.load('models/new_20per_ju_only_512.pt')   
    model3 = torch.jit.load('models/bok_512_32_55.pt')   
    model4 = torch.jit.load('models/cohabit_512_16_100.pt')   
    #     model4 = torch.jit.load('models/true_beauty_2nd_female_only_ju_512_freezeD.pt')   
    model_list = [model1, model2, model3, model4]
    model_name_list = ['ch_tb_32', 'sk_tb_16', 'sk_tb_32', 'ch_freethrow']

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 6006)))
    