def paste_object(output_img_name, ball_number, human_number):
    output_img = cv2.imread(output_img_name)
    occlusion = 0
    
    #Define how many objects to paste and which type of object each one is
    objects = []
    for j in range(ball_number):
        objects.append(1)
    for k in range(human_number):
        objects.append(0)

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
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            
            #Create a view-specific copy-paste area
            if '_0:' in output_img_name: #Left view
                random_x = random.randint(dst_w, dst_w//5)
                random_y = random.randint(dst_h//2 - dst_h//5, dst_h//2 + dst_h//5)
            else: #Right view
                random_x = random.randint(0, dst_w - dst_w//5)
                random_y = random.randint(dst_h//2 - dst_h//5, dst_h//2 + dst_h//5)
            
            x_c = random_x
            y_c = random_y
            
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
            
            #Calculate center coords of the output image
            # x_c, y_c = dst_w // 2, dst_h // 2
            
            #Create random coords from the center of the image for the new object
            # random_h = random.randint(-200, 200)
            # random_w = random.randint(-400, 400)
            
            #Update x_c and y_c with the random values
            # x_c = x_c + random_wimage_path = 'Data/src_images/0_aug/KS-FR-BLOIS_24330_1513711368928_1:1.png'
            # y_c = y_c + random_h
            
            #Change the polygon pairs coords to adapt it to the new center, create a mask for destination image and fill it with new pairs
            for p in poly:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(output_img.shape, output_img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            
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
            
    
    #Create and write the new annotation file
    with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt', 'w') as f:
        for dst_point in dst_points:
            f.write(f'{dst_point}\n')
    
    #Create the png mask 
    #convert_text_to_png(f'{data_dir}/{aug_label_dir}/{os.path.basename(output_img_name)[:-4]}.txt', f'{data_dir}/{mask_dir}/{os.path.basename(output_img_name)[:-4]}.png', dst_h, dst_w)
    
    #Write the new image
    cv2.imwrite(f'{data_dir}/{aug_image_dir}/{os.path.basename(output_img_name)}', output_img)


def paste_on_images(augmented_path, number_of_images):
    kont = 1
    brk = 0
    for i in range(18):
        path = augmented_path + '/' + str(i) + '_aug/*.png'
        image_files = glob.glob(path)
        for image_file in image_files:
            #Decide how many balls and humans to be pasted for each image
            ball_number = 2
            human_number = 4
            paste_object(image_file, ball_number, human_number)
            if kont == number_of_images:
                brk = 1
                break
            kont = kont + 1
        if brk == 1:
            break
