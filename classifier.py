from keras import backend as K
K.set_image_dim_ordering('tf')

import cv2, os, numpy, sys
import pandas as pd
import multiprocessing
import time

from keras.models import model_from_json
from joblib import Parallel, delayed


numpy.random.seed(1337)

def processPatches(data, patch_size, pixel_step, resize, nb_channels):
    # get data
    section_mat = data.values
    half_patch = int(patch_size/2)
    
    # get image info
    nb_rows = data.shape[0] 
    nb_cols = data.shape[1] 
    
    count_patches = 0
    patch_name_list = []
    patch_list = []
    for i in range (half_patch, nb_rows - half_patch, pixel_step):
        for j in range (half_patch, nb_cols - half_patch, pixel_step):
            # create patch
            start_row = i - half_patch
            start_col = j - half_patch
            patch =  numpy.zeros((patch_size,patch_size)) # 1 empty patch
            for x in range(patch_size):
                for y in range(patch_size):
                    patch[x][y] = section_mat[start_row + x][start_col + y]
            # resize, clip
            patch = cv2.resize(patch, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            patch = numpy.clip(patch, -1., 1.)
            # append to global list
            patch_list.append(patch)
            patch_name = 'patch_p_' + str(i) + '_' + str(j) + '.csv'
            patch_name_list.append(patch_name)
            # count
            count_patches +=1
    
    return patch_list, patch_name_list

def classify(input_dir, patch_size, pixel_step, jsonModelFilePath, weightsFilePath, modelName):
    start_time = time.time()
    
    # set params
    resize = 45 
    imageChannels = 1
    
    # create output directory
    directory = "output/classification/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # read all files in 1 step
    df_list = []
    df_names = []
    files = os.listdir(input_dir)
    for i in range(0, len(files)):
        filename = files[i]
        section_name = filename.split('_')[0]
        df = pd.read_csv(input_dir + filename, delimiter=' ', header = None)
        df_list.append(df)
        df_names.append(modelName + "_" + section_name)
        
    # load model
    jsonModelFile = open(jsonModelFilePath, 'r' )
    jsonModel = jsonModelFile.read()
    jsonModelFile.close()
    model = model_from_json(jsonModel)
    model.load_weights(weightsFilePath)
    model.compile( loss='binary_crossentropy', optimizer='sgd', metrics=[ 'accuracy' ] )
    
    # prepare save prediction for all sections
    nb_sections = len(df_list)
    
    # create patches in parrallel 
    s = 0
    nb_section = len(df_list)
    section_step = 4
    while 1 :
        s_init = s
        
        if(s == nb_section):
            break
        
        if(s+section_step > len(df_list)):
            df_sub_list = df_list[s:len(df_list)]
            s = nb_section
        else:
            df_sub_list = df_list[s:s+section_step]
            s = s+section_step
        
        print("Creating patches...")
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose = 100)(delayed(processPatches)(i,  patch_size, pixel_step, resize, imageChannels) for i in df_sub_list)
        current_sections_patch_lists, current_sections_patch_name_lists = zip(*results)
    
        # classify and save
        current_sections_prediction_lists = []
        for i in range(0, len(current_sections_patch_lists)):
            print("Classifying section " + str(s_init + i + 1) + "/" + str(nb_section))
            patch_list = current_sections_patch_lists[i]
            patches = numpy.array( patch_list ) 
            patches = patches.reshape( patches.shape[0], resize, resize, imageChannels)
            patches = patches.astype( 'float32' )
            # classify
            classesPredictionList = []
            classesPredictionList = model.predict_classes(patches)
            current_sections_prediction_lists.append(classesPredictionList)

        print("Writing classification files...")
        for i in range(0, len(current_sections_patch_lists)):
            print("Section " + df_names[s_init + i])
            predictionsFile = open(directory + 'classification_' + df_names[s_init + i] + '.txt', 'w')
            for j in range(0, len(current_sections_prediction_lists[i])):
                patch_name = current_sections_patch_name_lists[i][j]
                prediction = current_sections_prediction_lists[i][j]
                predictionsFile.write( patch_name + " " + str(prediction) + "\n" )
            predictionsFile.close()
               
    print("--- %s seconds ---" % (time.time() - start_time))
    
def classifySVM(input_dir, patch_size, pixel_step, jsonModelFilePath, weightsFilePath, modelName, svmModelPath):
    start_time = time.time()

    # set params
    resize = 45 # todo = read from json model
    imageChannels = 1
    
    # create output directory
    directory = "output/classification/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # read all files in 1 step
    df_list = []
    df_names = []
    files = os.listdir(input_dir)
    for i in range(0, len(files)):
        filename = files[i]
        section_name = filename.split('_')[0]
        df = pd.read_csv(input_dir + filename, delimiter=' ', header = None)
        df_list.append(df)
        df_names.append(modelName + "_" + section_name)
        
    # load CNN
    jsonModelFile = open(jsonModelFilePath, 'r' )
    jsonModel = jsonModelFile.read()
    jsonModelFile.close()
    model = model_from_json(jsonModel)
    model.load_weights(weightsFilePath)    
    #------ delete last layers -------
    for i in range (7):
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
    #------ Load SVM
    from sklearn.externals import joblib
    clf = joblib.load(svmModelPath) 
    
    # prepare save prediction for all sections
    nb_sections = len(df_list)
    
    # create patches in parrallel 
    s = 0
    nb_section = len(df_list)
    section_step = 4
    while 1 :
        
        s_init = s
        
        if(s == nb_section):
            break
        
        if(s+section_step > len(df_list)):
            df_sub_list = df_list[s:len(df_list)]
            s = nb_section
        else:
            df_sub_list = df_list[s:s+section_step]
            s = s+section_step
        
        print("Creating patches...")
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose = 100)(delayed(processPatches)(i,  patch_size, pixel_step, resize, imageChannels) for i in df_sub_list)
        current_sections_patch_lists, current_sections_patch_name_lists = zip(*results) # returns tuples :/
    
    
        # classify and save
        current_sections_prediction_lists = []
        for i in range(0, len(current_sections_patch_lists)):
            print("Classifying section " + str(s_init + i + 1) + "/" + str(nb_section))
            patch_list = current_sections_patch_lists[i]
            patches = numpy.array( patch_list ) 
            patches = patches.reshape( patches.shape[0], resize, resize, imageChannels)
            patches = patches.astype( 'float32' )
            # classify
            classesPredictionList = []
            features = model.predict(patches)
            classesPredictionList = clf.predict(features)
            current_sections_prediction_lists.append(classesPredictionList)

        print("Writing classification files...")
        for i in range(0, len(current_sections_patch_lists)):
            print("Section " + df_names[s_init + i])
            predictionsFile = open(directory + 'classification_' + df_names[s_init + i] + '.txt', 'w')
            for j in range(0, len(current_sections_prediction_lists[i])):
                patch_name = current_sections_patch_name_lists[i][j]
                prediction = current_sections_prediction_lists[i][j]
                predictionsFile.write( patch_name + " " + str(prediction) + "\n" )
            predictionsFile.close()
        
    print("--- %s seconds ---" % (time.time() - start_time))



