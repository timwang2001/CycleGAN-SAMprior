from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# def get_sam_mask(img)
#     sam = sam_model_registry["default"](checkpoint="pretrained_model/sam_vit_h_4b8939.pth").to('cuda')
#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=32,
#         points_per_batch=16,
#         pred_iou_thresh=0.88,
#         stability_score_thresh=0.92,
#         crop_n_layers=1,
#         crop_n_points_downscale_factor=2,
#         min_mask_region_area=100,  
#     )
#     bgrimg = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
def get_sam_mask(image):
    sam = sam_model_registry["default"](checkpoint="pretrained_model/sam_vit_h_4b8939.pth").to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=16,#调大了会快点
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    image1 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    masks = mask_generator.generate(image1)
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    img=img*255
    img=img.astype(np.uint8)
    img_image=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGBA2RGB))
    img_image=img_image.convert('L')
    return img_image

# def show_anns(anns):#返回的就是要的PIL的mask
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     img=img[:,:,:3]
#     img=img*255
#     img=img.astype(np.uint8)
#     img_image=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#     #print(type(img_image))
#     #img_image.save("image_save_20200509.jpg")
#     return img_image
#     #ax.imshow(img)

# image = Image.open('031.jpg')
# image1 = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
# masks = mask_generator.generate(image1)
# #cv2.imwrite('00.jpg',mask[0])
# #plt.figure(figsize=(w/77,h/77))
# #plt.imshow(image)
# img_PIL=show_anns(masks)