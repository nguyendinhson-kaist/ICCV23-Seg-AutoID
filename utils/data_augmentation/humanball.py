import cv2
import numpy as np
import os
import random

def find_extreme_points(points):
    # Initialize the extreme points
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    for point in points:
        x, y = point

        # Find the top-left corner
        if top_left is None or (x <= top_left[0] and y <= top_left[1]):
            top_left = (x, y)

        # Find the top-right corner
        if top_right is None or (x >= top_right[0] and y <= top_right[1]):
            top_right = (x, y)

        # Find the bottom-left corner
        if bottom_left is None or (x <= bottom_left[0] and y >= bottom_left[1]):
            bottom_left = (x, y)

        # Find the bottom-right corner
        if bottom_right is None or (x >= bottom_right[0] and y >= bottom_right[1]):
            bottom_right = (x, y)

    return top_left, top_right, bottom_right, bottom_left


def human_ball_interaction(file):
    
    # Read the image
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read the annotations
    annotation_path = file.replace('new_images', 'new_labels')
    annotation_path = annotation_path.replace('.png', '.txt')
    with open(annotation_path, 'r') as f:
        content = f.read()
        
    # Split the annotation into different paragraphs
    paragraphs = content.split('\n\n')
    for paragraph in paragraphs:
        paragraph.strip()
        lines = paragraph.split('\n')
        for line in lines:
            if not line=='':
                # Split the annotation into individual elements
                elements = line.split()
                
                if elements[0] == 'human':
                    # Extract the class label and coordinates
                    coordinates = [int(coord) for coord in elements[1:]]
                    pairs = [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]
                    top_left, top_right, bottom_right, bottom_left = find_extreme_points(pairs)
                    top_left = (int(top_left[0]), int(top_left[1]))
                    top_right = (int(top_right[0]), int(top_right[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                    #cv2.polylines(image, [np.array([top_left, top_right, bottom_right, bottom_left], np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    # Choose where to paste the ball, in this case the top-right
                    y_c = top_right[1]
                    x_c = top_right[0]
                    
                    # Take a ball
                    weights = [0.3, 0.7]
                    values = [0,1] #0 means normal ball and 1 pure ball
                    random_number_ball = random.choices(values, weights)[0]
                    if random_number_ball == 0:
                        random_number_ball = random.randint(1, 120)
                    else:
                        random_number_ball = random.randint(120, 127)
                    
                    src_name = f'../../data/c_images/1/ball_{random_number_ball}.png'
                    with open('../../data/c_labels/1.txt') as f:
                        labels = {}
                        for line in f.readlines():
                            line = line.rstrip().split(' ')
                            labels[line[0]] = line[1:]
                    label = labels[os.path.basename(src_name)]
                    src_img = cv2.imread(src_name)
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                    
                    #Divide the label into number pairs
                    poly = []
                    dst_poly = []
                    dst_points = []
                    for p in range(0, len(label), 2):
                        poly.append([int(label[p]), int(label[p + 1])])
                    
                    #Create a mask and fill it with the previous pairs
                    src_mask = np.zeros(src_img.shape, src_img.dtype)
                    cv2.fillPoly(src_mask, [np.array(poly)], (255, 255, 255))
                    
                    #Get the dimensions of the object image
                    h, w = image.shape[:2]
                    obj_h, obj_w = src_img.shape[:2]
                    
                    #Change the polygon pairs coords to adapt it to the new center, create a mask for destination image and fill it with new pairs
                    for p in poly:
                        dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
                    dst_mask = np.zeros(image.shape, image.dtype)
                    cv2.fillPoly(dst_mask, [np.array(dst_poly, int)], (255, 255, 255))
                    dst_point = ['ball']
                    
                    #Create the annotation by adding the pairs
                    for p in dst_poly:
                        dst_point.append(p[0])
                        dst_point.append(p[1])
                    dst_point = " ".join([str(p) for p in dst_point])
                    dst_points.append(dst_point)
                    
                    #Remove the part of the original image corresponding to the new pasted object
                    image[dst_mask > 0] = 0
                    
                    #Add the new object to the corresponding part of the output image
                    image[y_c:y_c + obj_h, x_c:x_c + obj_w] = image[y_c:y_c + obj_h, x_c:x_c + obj_w] + src_img * (src_mask > 0)
    
    # Check previous annotations for the image
    # annotation_path = f'{data_dir}/{source_label_dir}/{os.path.basename(file)[:-4]}.txt'
    # if not os.path.exists(annotation_path):
    #     #Create and write the new annotation file
    #     with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(file)[:-4]}.txt', 'w') as f:
    #         for dst_point in dst_points:
    #             f.write(f'{dst_point}\n')
    # else:
    #     with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(file)[:-4]}.txt', 'w') as f, open(annotation_path, 'r') as a:
    #         f.write(a.read())
    #         f.write('\n')
    #         for dst_point in dst_points:
    #             f.write(f'{dst_point}\n')
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('testimages/humanballtest.png', image)
    

human_ball_interaction('../../data/new_images/KS-FR-BLOIS_24330_1513710529019_0:1.png')