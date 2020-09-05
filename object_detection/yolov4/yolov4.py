# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:17:12 2020

@author: ppike
"""


import sys,os, paramiko
from pathlib import Path

sys.path.append(os.path.join(Path(os.path.realpath(__file__)).parent))

import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd


class YoloV4:
    
    def __init__(self, framework='tf', size=608, tiny=False, model='yolov4', NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES)),load_h5 = False, h5_file= None):
        self.framework = framework
        self.weights = 'weights' #None#weights
        self.size = size
        self.tiny = tiny
        self.model = model
        self.instanciated_model = None
        
        # Instanciate model
        
        if load_h5:
            print('Loading Model from h5 file')
            self.instanciated_model = tf.keras.models.load_model(h5_file)
        
        else:
            print('Tiny ',self.tiny)
    
            #image_path = self.image
            #NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
            input_size = self.size
            if self.framework == 'tf':
                input_layer = tf.keras.layers.Input([input_size, input_size, 3])
                if self.tiny:
                    if self.model == 'yolov3':
                        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
                        self.weights = os.path.join(self.weights,'yolov3-tiny.weights')
                    else:
                        feature_maps = YOLOv4_tiny(input_layer, NUM_CLASS)
                        self.weights = os.path.join(self.weights,'yolov4-tiny.weights')
                    bbox_tensors = []
                    for i, fm in enumerate(feature_maps):
                        bbox_tensor = decode(fm, NUM_CLASS, i)
                        bbox_tensors.append(bbox_tensor)
                    model = tf.keras.Model(input_layer, bbox_tensors)
                    model.summary()
                    
                    ##Added
                    if self.weights.split(".")[len(self.weights.split(".")) - 1] == "weights":
                        print('test_0')
                        if self.model == 'yolov3':
                            utils.load_weights_tiny(model, self.weights, 'yolov3')
                        else:
                            utils.load_weights_tiny(model, self.weights, 'yolov4')
                    else:
                        print('test_1')
                        model.load_weights(self.weights).expect_partial()
                        
                    #utils.load_weights_tiny(model, self.weights, self.model)
                    ##
                else:
                    if self.model == 'yolov3':
                        feature_maps = YOLOv3(input_layer, NUM_CLASS)
                        bbox_tensors = []
                        for i, fm in enumerate(feature_maps):
                            bbox_tensor = decode(fm, NUM_CLASS, i)
                            bbox_tensors.append(bbox_tensor)
                        model = tf.keras.Model(input_layer, bbox_tensors)
                        
                        yolov3_weights_path = os.path.join(self.weights,'yolov4.weights')
                        #utils.load_weights_v3(model, self.weights)
                    elif self.model == 'yolov4':
                        feature_maps = YOLOv4(input_layer, NUM_CLASS)
                        bbox_tensors = []
                        for i, fm in enumerate(feature_maps):
                            bbox_tensor = decode(fm, NUM_CLASS, i)
                            bbox_tensors.append(bbox_tensor)
                        model = tf.keras.Model(input_layer, bbox_tensors)
                        
                        # Check if files have already been downloaded
                        yolov4_weights_path = os.path.join(self.weights,'yolov4.weights')
                        #yolov4_weights_path = os.path.join(Path(os.path.realpath(__file__)).parent,'data/yolov4.weights')
                        
                        if not os.path.exists(yolov4_weights_path):
                            print('Downloading weights file')
                            self.weights = self.download('yolov4.weights',local_path=self.weights)
                            print('Weight file was downloaded to',self.weights)
                        else:
                            print('Weights file already downloaded')
                            self.weights = yolov4_weights_path
        
                        if self.weights.split(".")[len(self.weights.split(".")) - 1] == "weights":
                            if self.model == 'yolov3':
                                utils.load_weights(model, yolov3_weights_path)
                            elif self.model == 'yolov4':
                                utils.load_weights(model, self.weights)
                        else:
                            model.load_weights(self.weights).expect_partial()
                            
                self.instanciated_model = model
                
            else:
                # Load TFLite model and allocate tensors.
                interpreter = tf.lite.Interpreter(model_path=self.weights)
                interpreter.allocate_tensors()
    
                self.instanciated_model = interpreter
    
    def download(self, filename, remote_path='../home/paul/weights_files/', local_path='weigths/', hostname='46.101.102.251', username='root', password='KAisensdata.2020Kaisens'):
    
        # Connect to server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, username=username, password=password)
        
        
        # Load File
        src_path = remote_path+filename
        
        if not os.path.exists(local_path):
            os.mkdir(local_path)
            
        dst_path = os.path.join(local_path,filename)
        #dst_path = os.path.join(Path(os.path.realpath(__file__)).parent,'data/'+filename)
        
        ftp_client=ssh_client.open_sftp()
        ftp_client.get(src_path,dst_path)
        ftp_client.close()
        
        return dst_path
        
    def predict(self, image_path, result_dir='.', save_img=True, save_dataframe=True, image_name=None):
        
        #print('\ntype:',type(image_path))
        #try:
        if type(image_path) is not np.ndarray:
        
            if not(os.path.exists(image_path)):
                print('No such file or directory',image_path)
                
                #return None
            else:
                original_image = cv2.imread(image_path)
                #print('Shape1', original_image.shape)
        #except:
        else:
            original_image = image_path
            #print('Shape2', original_image.shape)
        
        if self.tiny:
            STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, self.tiny)
            XYSCALE = cfg.YOLO.XYSCALE_TINY
        else:
            STRIDES = np.array(cfg.YOLO.STRIDES)
            if self.model == 'yolov4':
                ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, self.tiny)
            else:
                ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, self.tiny)
            XYSCALE = cfg.YOLO.XYSCALE
        
        input_size = self.size
        
        try:
            #print('image:',original_image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
        except:
            return pd.DataFrame()
    
        image_data = utils.image_preprocess(np.copy(original_image), [self.size,self.size])#[input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        if self.framework == 'tf':
            model = self.instanciated_model
            #model.summary()
            pred_bbox = model.predict(image_data)
            
        else:
            interpreter = self.instanciated_model
            
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            
        if self.model == 'yolov4':
            if self.tiny:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE, RESIZE=1.5)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = utils.nms(bboxes, 0.213, method='nms')
    
        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        #image.show()
        
        classes = utils.read_class_names(cfg.YOLO.CLASSES)
        list_bboxes = []
        
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            #print('type bbox',type(bbox))
            #print('bbox',bbox[:4])
            #print('coor',list(coor))
            bbox_info = {'coor':list(coor),'probability':score,'class':classes[class_ind]}
            list_bboxes.append(bbox_info)

        if image_name == None:
            output_name = os.path.join(result_dir + '/out_' + os.path.basename(image_path))
        else:
            output_name = os.path.join(result_dir + '/out_' + str(image_name) + '.jpg')
            # output_name = os.path.join(result_dir+'/out_' + str(image_name) + '.jpg')
            # if save_img:
            # image.save(output_name)
            # cv2.imwrite(output_name,img)
            # print('Img saved to',output_name)
        try:
            # output_name = os.path.join(result_dir+'/out_' + os.path.basename(image_path))
            if save_img:
                image.save(output_name)
                # cv2.imwrite(output_name,img)
                print('Img saved to', output_name)

            output = pd.DataFrame(list_bboxes)
            if save_dataframe:
                # print('image_path',image_path )
                output_name = '.'.join(output_name.split('.')[:2]) + '.xlsx'
                # output_name = 'results/out_'+image_path.split('\\')[-1].split('.')[0]+'.xlsx'
                print('Result file saved to', output_name)
                output.to_excel(output_name)
            return output

        except Exception as e:
            print("exception for xlsx")
            print(e)
        
        return pd.DataFrame()


#yolo = YoloV4()
#yolo.predict('1fc35a5149379fff131e939f18257341.7.jpeg')

# Working Class

# =============================================================================
# class YoloV4:
#     
#     def __init__(self,framework = 'tf', weights=os.path.join(Path(os.path.realpath(__file__)).parent,'data/yolov4.weights'),size=608,tiny=False,model='yolov4'):
#         self.framework = framework
#         self.weights = weights
#         self.size = size
#         self.tiny = tiny
#         self.model = model
#         self.instanciated_model = None
#         
#         # Instanciate model
#         
#         print('Tiny ',self.tiny)
# 
#         #image_path = self.image
#         NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
#         input_size = self.size
#         if self.framework == 'tf':
#             input_layer = tf.keras.layers.Input([input_size, input_size, 3])
#             if self.tiny:
#                 if self.model == 'yolov3':
#                     feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
#                 else:
#                     feature_maps = YOLOv4_tiny(input_layer, NUM_CLASS)
#                 bbox_tensors = []
#                 for i, fm in enumerate(feature_maps):
#                     bbox_tensor = decode(fm, NUM_CLASS, i)
#                     bbox_tensors.append(bbox_tensor)
#                 model = tf.keras.Model(input_layer, bbox_tensors)
#                 model.summary()
#                 utils.load_weights_tiny(model, self.weights, self.model)
#             else:
#                 if self.model == 'yolov3':
#                     feature_maps = YOLOv3(input_layer, NUM_CLASS)
#                     bbox_tensors = []
#                     for i, fm in enumerate(feature_maps):
#                         bbox_tensor = decode(fm, NUM_CLASS, i)
#                         bbox_tensors.append(bbox_tensor)
#                     model = tf.keras.Model(input_layer, bbox_tensors)
#                     utils.load_weights_v3(model, self.weights)
#                 elif self.model == 'yolov4':
#                     feature_maps = YOLOv4(input_layer, NUM_CLASS)
#                     bbox_tensors = []
#                     for i, fm in enumerate(feature_maps):
#                         bbox_tensor = decode(fm, NUM_CLASS, i)
#                         bbox_tensors.append(bbox_tensor)
#                     model = tf.keras.Model(input_layer, bbox_tensors)
#     
#                     if self.weights.split(".")[len(self.weights.split(".")) - 1] == "weights":
#                         utils.load_weights(model, self.weights)
#                     else:
#                         model.load_weights(self.weights).expect_partial()
#                         
#                 self.instanciated_model = model
#             
#         else:
#             # Load TFLite model and allocate tensors.
#             interpreter = tf.lite.Interpreter(model_path=self.weights)
#             interpreter.allocate_tensors()
# 
#             self.instanciated_model = interpreter
#             
#         
#     def predict(self,image_path,result_dir='results',save_img=True):
#         
#         if self.tiny:
#             STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
#             ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, self.tiny)
#             XYSCALE = cfg.YOLO.XYSCALE_TINY
#         else:
#             STRIDES = np.array(cfg.YOLO.STRIDES)
#             if self.model == 'yolov4':
#                 ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, self.tiny)
#             else:
#                 ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, self.tiny)
#             XYSCALE = cfg.YOLO.XYSCALE
#         
#         input_size = self.size
#         
#         original_image = cv2.imread(image_path)
#         print('image:',original_image)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#         original_image_size = original_image.shape[:2]
#     
#         image_data = utils.image_preprocess(np.copy(original_image), [self.size,self.size])#[input_size, input_size])
#         image_data = image_data[np.newaxis, ...].astype(np.float32)
#         
#         if self.framework == 'tf':
#             model = self.instanciated_model
#             model.summary()
#             pred_bbox = model.predict(image_data)
#             
#         else:
#             interpreter = self.instanciated_model
#             
#             # Get input and output tensors.
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()
#             print(input_details)
#             print(output_details)
#             
#             interpreter.set_tensor(input_details[0]['index'], image_data)
#             interpreter.invoke()
#             pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
#             
#         if self.model == 'yolov4':
#             if self.tiny:
#                 pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE, RESIZE=1.5)
#             else:
#                 pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
#         else:
#             pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
#         bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
#         bboxes = utils.nms(bboxes, 0.213, method='nms')
#     
#         image = utils.draw_bbox(original_image, bboxes)
#         image = Image.fromarray(image)
#         #image.show()
#         
#         print('Image path',image_path)
#         print('Type Image path',type(image_path))
#         print('Bboxes type',type(bboxes))
#         
#         classes = utils.read_class_names(cfg.YOLO.CLASSES)
#         list_bboxes = []
#         
#         for i, bbox in enumerate(bboxes):
#             coor = np.array(bbox[:4], dtype=np.int32)
#             score = bbox[4]
#             class_ind = int(bbox[5])
#             #print('type bbox',type(bbox))
#             #print('bbox',bbox[:4])
#             #print('coor',list(coor))
#             bbox_info = {'coor':list(coor),'probability':score,'class':classes[class_ind]}
#             list_bboxes.append(bbox_info)
#         
#         try:
#             output_name = os.path.join(result_dir+'/out_' + os.path.basename(image_path))
#             
#             if save_img:
#                 image.save(output_name)
#                 #cv2.imwrite(output_name,img)
#                 print('Img saved to',output_name)
#             
#             output = pd.DataFrame(list_bboxes)
#             print('image_path',image_path )
#             output_name = '.'.join(output_name.split('.')[:2])+'.xlsx'
#             #output_name = 'results/out_'+image_path.split('\\')[-1].split('.')[0]+'.xlsx'
#             print('output_name',output_name)
#             output.to_excel(output_name)
#             
#         except Exception as e:
#             print(e)
# =============================================================================
            
#yolo = YoloV4()
#yolo.predict('1fc35a5149379fff131e939f18257341.7.jpeg')