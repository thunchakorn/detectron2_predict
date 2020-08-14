import cv2
import numpy as np

def visualising(image, results, save_name):
    image_layout = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in results:
        points = np.array(r['polygon']).astype(np.int).reshape(-1,1,2)
        image_layout = cv2.polylines(image_layout, [points], True, (0,255,0), 3)
        text_pos = tuple(map(lambda x: int(x), r['bbox'][:2]))
        image_layout = cv2.putText(image_layout,r['label'] + ' - ' + str(np.round(r['score'], 4)), text_pos, font, 1,(0,0,255), 4, cv2.LINE_AA)
        
    cv2.imwrite(save_name, image_layout)