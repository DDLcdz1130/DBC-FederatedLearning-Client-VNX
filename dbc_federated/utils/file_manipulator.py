import os
import ntpath
import sys
import glob
import random
import copy
import shutil
from shutil import copyfile


input_path = './data_wood/'
out_folders = ['./folder1/', './folder2/','./folder3/']
out_fractions = [0.3, 0.3, 0.4]
shuff = True

def general_split(input_path, out_folders, out_fractions, shuff = True, remove_original=False):
    #print(os.path.join(input_path, outputpath))
    #print(os.walk(input_path))

    for dirpath, dirnames, filenames in os.walk(input_path):
        print("dirpath:")
        print(dirpath)
        print("dirnames:")
        print(dirnames)
        
        n=len(filenames)
        print("number of files: "+str(n))
        lout= len(out_folders)
        k=0
        if shuff == True:
            random.shuffle(filenames)
        else:
            pass
        out_number = []

        for i, (outputpath, outfraction) in enumerate(zip(out_folders, out_fractions)):
            structure = os.path.join(outputpath, os.path.relpath(dirpath,input_path))
            print("structure:")
            print(structure)
            if not os.path.isdir(structure):
                os.mkdir(structure)
            else:
                print("Folder does already exits!")
            if i < lout-1:
                x = int(n*outfraction)
            elif out_number != []:
                x = n-sum(out_number)
            elif out_number == 0:
                x = 0
            out_number.append(x)
            
            for filename in filenames[k:(k+x)]:
                copyfile(os.path.join(dirpath,filename), os.path.join(structure, filename))

    if remove_original==True:
        shutil.rmtree(input_path)    









def refolder(data_folder, targ_folder, train_fraction=0.8, val_fraction=0.2, test_fraction=0.0, 
              remove_original=False):
    r=data_folder
    classes=[f for f in os.listdir(r) if os.path.isdir(os.path.join(r,f))]
    print('1 step')
    if os.path.isdir(targ_folder):
        shutil.rmtree(targ_folder)
    os.mkdir(targ_folder)
    print('step 2')
    sub_folder=os.path.join(targ_folder, 'train')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))
    
    sub_folder=os.path.join(targ_folder, 'val')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))

    if test_fraction!=0:
        sub_folder=os.path.join(targ_folder, 'test')
        os.mkdir(sub_folder)
        for c in classes:
            os.mkdir(os.path.join(sub_folder,c))
    
    for c in classes:
        files=glob.glob(os.path.join(r,c,"*"))
        random.shuffle(files)
        train_n=int(len(files)*train_fraction)
        for f in files[:train_n]:
            filename = os.path.basename(f)
            copyfile(f, os.path.join(targ_folder,'train', c,filename))
        
        if test_fraction==0:
            for f in files[train_n:]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
        
        elif test_fraction!=0:
            val_n=int(len(files)*val_fraction)
            for f in files[train_n:(train_n+val_n)]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
            for f in files[(train_n+val_n):]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'test', c,filename))
        
        if remove_original==True:
            shutil.rmtree(data_folder)
