# from super_gradients.training import models
# import torch
#
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# yolo_nas_l = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
#
# # Convert to ONNX
# models.convert_to_onnx(model=yolo_nas_l, input_shape=(3, 640, 640), out_path="yolo_nas_l.onnx")


# pip install roboflow
#
# from roboflow import Roboflow
# rf = Roboflow(api_key="TRWhm2GxKObXkuY6hQDs")
# project = rf.workspace("usc-sdm42").project("a-oeu4q")
# dataset = project.version(1).download("yolov8")
