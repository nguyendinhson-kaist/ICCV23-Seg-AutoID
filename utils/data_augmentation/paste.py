import glob
import os
import random
import cv2
import numpy as np
import albumentations as A
import json
import tqdm

data_dir = '../../data'
image_dir = 'images'
label_dir = 'labels'
crop_image_dir = 'c_images'
crop_label_dir = 'c_labels'
annotation_dir = 'annotations'
source_image_dir = 'src_images'
source_label_dir = 'src_labels'
aug_image_dir = 'new_images'
aug_label_dir = 'new_labels'
mask_dir = 'new_masks'
augmented_path = '../../data/src_images'

def convert2coco():
    
    # This function is in charge of converting all the augmented annotations in txt files into a single final json file. It doesn't have any input.
    
    print('Converting into COCO ...')
    classes = ('human', 'ball')
    filenames = [filename for filename in os.listdir(f'{data_dir}/{aug_image_dir}')]
    img_id = 0
    box_id = 0
    images = []
    categories = []
    annotations = []
    for filename in tqdm.tqdm(filenames):
        img_id += 1
        h, w = cv2.imread(f"{data_dir}/{aug_image_dir}/{filename}").shape[:2]
        images.append({'file_name': filename, 'id': img_id, 'height': h, 'width': w})
        regions = []
        with open(f'{data_dir}/{aug_label_dir}/{filename[:-4]}augmented.txt') as f:
            for line in f.readlines():
                regions.append(line.rstrip())
        for region in regions:
            box_id += 1
            region = region.split(' ')
            mask = region[1:]
            poly = []
            for i in range(0, len(mask), 2):
                poly.append([int(mask[i]), int(mask[i + 1])])
            x_min, y_min, w, h = cv2.boundingRect(np.array([poly], int))
            bbox = [x_min, y_min, w, h]

            category_id = classes.index(region[0]) + 1
            annotations.append({'id': box_id,
                                'bbox': bbox,
                                'iscrowd': 0,
                                'image_id': img_id,
                                'segmentation': [list(map(int, mask))],
                                'area': bbox[2] * bbox[3],
                                'category_id': category_id})
    for category_id, category in enumerate(classes):
        categories.append({'supercategory': category, 'id': category_id + 1, 'name': category})
    print(len(images), 'images')
    print(len(annotations), 'instances')
    json_data = json.dumps({'images': images, 'categories': categories, 'annotations': annotations})
    with open(f'{data_dir}/{annotation_dir}/train_aug.json', 'w') as f:
        f.write(json_data)


def convert_text_to_png(annotation, output_path, h, w):
    
    # This function is in charge of generating a png mask from its txt annotation. It takes 4 inputs:
    # -The txt annotation.
    # -The output path at which the mask will be stored.
    # -The height of the original image.
    # -The width of the original image.
    # As an output it stores the mask.
                        
    # Create a blank image
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    #Read the file
    with open(annotation, 'r') as f:
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
                # Extract the class label and coordinates
                coordinates = [int(coord) for coord in elements[1:]]
                # Draw the polygon on the image
                points = np.array(coordinates, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(image, [points], (255, 255, 255))
                
    # Save the image to the output path
    cv2.imwrite(output_path, image)
    
def convert_mask_to_txt(mask_path, output_path):
    
    # This function is in charge of converting a binary mask back to the txt file. It has two intpus:
    # -The path of the binary mask.
    # -The path of the output txt file.
    # As an output in writes the txt file with annotations.
    
    with open(output_path, 'a') as f:
        if 'human' in mask_path:
            f.write('human ')
        else:
            f.write('ball ')
        # Load the binary mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Find the white pixels (non-zero values) in the mask
        white_pixels = np.where(mask == 255)
        
        # Extract the x and y coordinates of the white pixels
        x_coords = white_pixels[1]
        y_coords = white_pixels[0]
        
        # Combine the x and y coordinates into pairs
        coordinates = [f"{x} {y}" for x, y in zip(x_coords, y_coords)]
        
        # Write the coordinates to a text file
        f.write(' '.join(coordinates))
        f.write('\n')

def average_close_points(points, threshold):
    
    # This function is in charge of averaging the intersections for defining the pasting area. It takes two inputs:
    # -A list with all the intersection points.
    # -A threshold to determine the minimum distance at which intersections can be.
    # As an output it returns the same list as input, but with the points that are too close to each other averaged into one.
    
    averaged_points = []
    while len(points) > 0:
        current_point = points[0]
        close_points = [current_point]
        remaining_points = []

        for point in points[1:]:
            distance = np.linalg.norm(np.array(current_point) - np.array(point))
            if distance <= threshold:
                close_points.append(point)
            else:
                remaining_points.append(point)

        averaged_point = np.mean(np.array(close_points), axis=0)
        averaged_points.append(averaged_point.tolist())
        points = remaining_points

    return averaged_points

def define_pasting_area(image_file, total_objects):
    
    # This function is in charge of defining the pasting area for the objects. It takes two inputs:
    # -The path to the image on which objects are gonna be pasted.
    # -The number of objects that will be pasted on it.
    # As an output it returns a list with all the coords such as coords = [(x1,y1), (x2,y2),...]
    
    # Read the image and get its dimensions
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Check if the image is right or left side of the court
    if '_0.png' in image_file:
        right = 0
    else:
        right = 1
    
    # Threshold the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(image_gray, 120, 150, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contour the thresholded image and filter the contours to get only the court
    filtered_contours = []
    blank_image = np.zeros_like(image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100000:  # Adjust the threshold as needed
            filtered_contours.append(contour)
    cv2.drawContours(blank_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Apply Hough Line Transform to detect straight lines
    gray = cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

    # Draw the detected lines on the original image
    min_line_length = 300
    blank_image_2 = np.zeros_like(image)
    enlarged_lines = []
    scale_factor = 3.5
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > min_line_length:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                dx = x2 - x1
                dy = y2 - y1

                # Compute the new endpoint coordinates
                new_x1 = center_x - int(dx * scale_factor / 2)
                new_y1 = center_y - int(dy * scale_factor / 2)
                new_x2 = center_x + int(dx * scale_factor / 2)
                new_y2 = center_y + int(dy * scale_factor / 2)

                # Append the enlarged line to the list of enlarged lines
                enlarged_lines.append([[new_x1, new_y1, new_x2, new_y2]])
                
    for line in enlarged_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank_image_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Find intersections of line segments
    filtered_intersections = []
    if enlarged_lines is not None:
        for i in range(len(enlarged_lines)):
            for j in range(i+1, len(enlarged_lines)):
                x1, y1, x2, y2 = enlarged_lines[i][0]
                x3, y3, x4, y4 = enlarged_lines[j][0]
                
                m1 = (y2-y1)/(x2-x1)
                m2 = (y4-y3)/(x4-x3)
                
                if m1 != m2: #The lines intersect
                    intersect_x = ((m1 * x1) - (m2 * x3) + y3 - y1) / (m1 - m2)
                    intersect_y = m1 * (intersect_x - x1) + y1
                    if x1 < intersect_x and intersect_x < x2 and y1 < intersect_y and intersect_y < y2: #the intersection is inside the lines
                        
                        # Get the direction vectors of the lines
                        v1 = (x2 - x1, y2 - y1)
                        v2 = (x4 - x3, y4 - y3)
                        
                        # Calculate the magnitudes of the direction vectors
                        v1_magnitude = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                        v2_magnitude = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
                        
                        # Normalize the direction vectors
                        v1_normalized = (v1[0] / v1_magnitude, v1[1] / v1_magnitude)
                        v2_normalized = (v2[0] / v2_magnitude, v2[1] / v2_magnitude)

                        # Calculate the dot product of the normalized vectors
                        dot_product = v1_normalized[0] * v2_normalized[0] + v1_normalized[1] * v2_normalized[1]

                        # Calculate the intersection angle in radians
                        angle_radians = np.arccos(dot_product)

                        # Convert the angle to degrees
                        angle_degrees = np.degrees(angle_radians)
                        if angle_degrees > 25:
                            filtered_intersections.append((intersect_x, intersect_y))
                
    filtered_intersections = average_close_points(filtered_intersections, 80)
    
    for point in filtered_intersections:
        intersect_x, intersect_y = point
        cv2.circle(blank_image_2, (int(intersect_x), int(intersect_y)), radius=10, color=(0, 0, 255), thickness=-1)
    
    gray = cv2.cvtColor(blank_image_2, cv2.COLOR_RGB2GRAY)
    # Apply Hough Line Transform to detect straight lines, to get the relative coords of the enlarged lines
    new_lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    blank_image_3 = np.zeros_like(image)
    for line in new_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank_image_3, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Get the points most to the right or left 
    extreme_points = []
    for line in new_lines:
        x1, y1, x2, y2 = line[0]
        if right == 1:
            if x1 == 0:
                extreme_points.append((x1,y1))
        else:
            if x2 == w-1:
                extreme_points.append((x2,y2))
    
    # Compare the points to get the topmost and bottomost
    topmost_point = None
    bottommost_point = None
    for point in extreme_points:
        x, y = point
        if topmost_point is None or y < topmost_point[1]:
            topmost_point = (x, y)
        if bottommost_point is None or y > bottommost_point[1]:
            bottommost_point = (x, y)
    
    # Apply thresholds on the two points
    x, y = topmost_point
    if y < h//3:
        topmost_point = (x, h//3)
    elif y > h//2:
        topmost_point = (x, h//2)
    x, y = bottommost_point
    if y > 5*h//6:
        bottommost_point = (x, 5*h//6)
    elif y < 3*h//4:
        bottommost_point = (x, 3*h//4)
       
    cv2.circle(blank_image_2, topmost_point, radius=10, color=(0, 0, 255), thickness=-1)
    cv2.circle(blank_image_2, bottommost_point, radius=10, color=(0, 0, 255), thickness=-1)
    
    # Get top and bot intersections
    top_inter = None
    bot_inter = None
    for point in filtered_intersections:
        x, y = point
        if top_inter is None or y < top_inter[1]:
            top_inter = (x, y)
        if bot_inter is None or y > bot_inter[1]:
            bot_inter = (x, y)
    
    if right == 1:
        final_points = [topmost_point, top_inter, bot_inter, bottommost_point]
        min_x = 0
        max_x = bot_inter[0]
        min_y = bottommost_point[1]
        max_y = top_inter[1]
    else:
        final_points = [top_inter, topmost_point, bottommost_point, bot_inter]
        min_x = bot_inter[0]
        max_x = w
        min_y = bottommost_point[1]
        max_y = top_inter[1]
    
    final_points = np.array(final_points, dtype=np.int32)
    cv2.polylines(image, [np.array(final_points)], isClosed=True, color=(255, 0, 0), thickness=6)
    
    
    # Create pasting area
    hull = cv2.convexHull(final_points)
    cont = 0
    finish = False
    paste_coords = []
    while not finish:
        point_inside = False
        while not point_inside:
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)
            distance = cv2.pointPolygonTest(hull, (random_x, random_y), measureDist=False)
            if distance > 0:
                point_inside = True
        random_point = (int(random_x), int(random_y))
        paste_coords.append(random_point)
        cont = cont + 1
        if cont == total_objects:
            finish = True
    
    return paste_coords
        

def apply_random_augmentation(image_path):
    
    # This function is in charge of applying a random augmentation to an image and its mask. The only input is:
    # -The path to the image to be augmented.
    # As an output it performs augmentation on an image and its mask and stores the masks separately in a folder.
    
    # Open image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    #Get output image dimensions
    h, w = input_image.shape[:2]
    
    # Open annotations
    label_path = image_path.replace('new_images', 'new_labels')
    label_path = label_path.replace('.png', '.txt')
    with open(label_path, 'r') as f:
        label = f.read()
    
    # Create a folder to store masks for each instance
    folder_path = label_path.replace('.txt', '')
    folder_path = folder_path.replace('new_labels', 'new_masks')
                
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
    
    # Create a list to store all the masks
    masks = []
    classes = []
    
    # Define transform
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    
    # Split the annotation into different paragraphs
    paragraphs = label.split('\n\n')
    for paragraph in paragraphs:
        paragraph.strip()
        lines = paragraph.split('\n')
        for line in lines:
            image = np.zeros((h, w, 3), dtype=np.uint8)
            if not line=='':
                
                # Split the annotation into individual elements
                elements = line.split()
                
                # Extract the class label and coordinates
                clas = elements[0]
                coordinates = [int(coord) for coord in elements[1:]]
                
                # Draw the polygon on the image
                points = np.array(coordinates, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(image, [points], (255, 255, 255))
                
                # Save masks
                masks.append(image)
                classes.append(clas)
                
    # Apply transformations
    transformed = transform(image=input_image, masks=masks)
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']
    
    # Save new image
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, transformed_image)
    
    # Save new masks
    cont = 0
    merged_mask = np.zeros_like(transformed_image)
    for transformed_mask in transformed_masks:
        mask_path = os.path.join(folder_path, 'mask_%d_%s.png' %(cont, classes[cont]))
        cv2.imwrite(mask_path, transformed_mask)
        # Update the txt annotations
        updated_txt_path =label_path.replace('.txt', 'augmented.txt') 
        convert_mask_to_txt(mask_path, updated_txt_path)
        cont = cont + 1
        merged_mask = merged_mask + transformed_mask
    mask_path = image_path.replace('new_images', 'new_masks')
    cv2.imwrite(mask_path, merged_mask)
    

def paste_object(output_img_name, ball_number, human_number):
    
    # This function is in charge of pasting all the objects to one image. The inputs are:
    # -The path of the image on which images are pasted
    # -The number of balls to be pasted.
    # -The number of humans to be pasted.
    # As an output, it saves the augmented image, its annotations in txt format and its mask.
    
    output_img = cv2.imread(output_img_name)
    occlusion = 0
    
    #Define how many objects to paste and which type of object each one is
    total_objects = ball_number + human_number
    objects = []
    for j in range(ball_number):
        objects.append(1)
    for k in range(human_number):
        objects.append(0)
    
    # Get the left-bottom corner coords of the objects to be pasted
    coords = define_pasting_area(output_img_name, total_objects)

    #Create a list for the annotations
    dst_points = []
    
    #If the image is not augmented
    if '0_aug' not in output_img_name:
        filename = os.path.basename(output_img_name).split(':')[0]
        with open(f'{data_dir}/{label_dir}/{filename}.txt') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                point = list(map(int, line[1:]))
                point.insert(0, line[0])
                point = " ".join([str(p) for p in point])
                dst_points.append(point)
                
    #If the image is augmented 
    else:
        for i in range(ball_number+human_number):
            #Get output image dimensions
            dst_h, dst_w = output_img.shape[:2]

            #Create empty lists for polygons
            poly = []
            dst_poly = []
    
            #Decide if the object to paste is a human or a ball, get a random object of that class and its label
            if objects[i] == 0:
                random_number_human = random.randint(1, 1897)
                src_name = f'{data_dir}/{crop_image_dir}/0/human_{random_number_human}.png'
                with open(f'{data_dir}/{crop_label_dir}/0.txt') as f:
                    labels = {}
                    for line in f.readlines():
                        line = line.rstrip().split(' ')
                        labels[line[0]] = line[1:]
            else:
                #Change the probabilities to generate more pure balls
                weights = [0.3, 0.7]
                values = [0,1] #0 means normal ball and 1 pure ball
                random_number_ball = random.choices(values, weights)[0]
                if random_number_ball == 0:
                    random_number_ball = random.randint(1, 120)
                else:
                    random_number_ball = random.randint(120, 127)
                
                src_name = f'{data_dir}/{crop_image_dir}/1/ball_{random_number_ball}.png'
                with open(f'{data_dir}/{crop_label_dir}/1.txt') as f:
                    labels = {}
                    for line in f.readlines():
                        line = line.rstrip().split(' ')
                        labels[line[0]] = line[1:]
            label = labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            
            #Divide the label into number pairs
            for p in range(0, len(label), 2):
                poly.append([int(label[p]), int(label[p + 1])])
            
            #Create a mask and fill it with the previous pairs
            src_mask = np.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [np.array(poly)], (255, 255, 255))
            
            # Get the left-corner coord of the object to be pasted and convert to center points
            x_corner, y_corner = coords[i]
            obj_w, obj_h = src_img.shape[:2]
            x_c = x_corner + obj_w//2
            y_c = y_corner - obj_h//2
            
            
            #Apply previously generated occlusion
            if occlusion == 1:
                if oclu_type != objects[i]:
                    x_c = oclu_x
                    y_c = oclu_y
                    occlusion = 2
            
            #Generate occlusion
            if occlusion == 0:
                occlusion_prob = 0.2
                prob = random.randint(1, 100)/100
                if prob < occlusion_prob:
                    oclu_x = x_c
                    oclu_y = y_c
                    oclu_type = objects[i]
                    occlusion = 1
            
            #Change the polygon pairs coords to adapt it to the new center, create a mask for destination image and fill it with new pairs
            for p in poly:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = np.zeros(output_img.shape, output_img.dtype)
            cv2.fillPoly(dst_mask, [np.array(dst_poly, int)], (255, 255, 255))
            
            #Get the dimensions of the object image
            h, w = src_img.shape[:2]
            
            #Write human or ball depending on the pasted object
            if objects[i] == 0: 
                dst_point = ['human']
            else:
                dst_point = ['ball']
            
            #Create the annotation by adding the pairs
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            dst_point = " ".join([str(p) for p in dst_point])
            dst_points.append(dst_point)
            
            #Remove the part of the original image corresponding to the new pasted object
            output_img[dst_mask > 0] = 0
            
            #Add the new object to the corresponding part of the output image
            output_img[y_c:y_c + h, x_c:x_c + w] += src_img * (src_mask > 0)
    
    # Check previous annotations for the image
    annotation_path = f'{data_dir}/{source_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt'
    if not os.path.exists(annotation_path):
        #Create and write the new annotation file
        with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt', 'w') as f:
            for dst_point in dst_points:
                f.write(f'{dst_point}\n')
    else:
        with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt', 'w') as f, open(annotation_path, 'r') as a:
            f.write(a.read())
            f.write('\n')
            for dst_point in dst_points:
                f.write(f'{dst_point}\n')
    
    #Create the png mask 
    convert_text_to_png(f'{data_dir}/{aug_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt', f'{data_dir}/{mask_dir}/{os.path.basename(output_img_name)[:-4]}.png', dst_h, dst_w)
    
    #Write the new image
    cv2.imwrite(f'{data_dir}/{aug_image_dir}/{os.path.basename(output_img_name)}', output_img)


def paste_on_images(augmented_path, number_of_images, ball_number, human_number):
    
    # This function takes the root directory at which images are stored for copy_paste augmentation. It also takes some hyperparameters: 
    # -The number of images to create.
    # -The number of humans to be pasted on each image.
    # -The number of balls to be pasted on each image.
    # It has no output, as the inner functions are in charge to store the augmented images.
    
    kont = 1
    brk = 0
    for i in range(18):
        path = augmented_path + '/' + str(i) + '_aug/*.png'
        image_files = glob.glob(path)
        for image_file in image_files:
            paste_object(image_file, ball_number, human_number)
            image_file = image_file.replace('src_images/0_aug', 'new_images')
            apply_random_augmentation(image_file)
            if kont == number_of_images:
                brk = 1
                break
            kont = kont + 1
        if brk == 1:
            break
    
def perform_copy_paste_and_random_augmentation(augmented_path, number_of_images):
    
    # This is the main function that call to the other functions. It has two inputs:
    # -The folder on which images for augmentation are.
    # -The number of images to create.
    # As an output it perfoms copy-paste and random augmentation into those images.
    # It has two hyperparameters: ball_number and human_number, to define how many objects to the pasted on each image.
    
    ball_number = 2
    human_number = 4
    paste_on_images(augmented_path, number_of_images, ball_number, human_number)
    convert2coco()