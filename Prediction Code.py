#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import radiomics
import radiomics.featureextractor as FEE
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy.stats import entropy
import math
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


# In[2]:


# Load images.
def load_nii_file(path):
    nii_img = nib.load(path)
    nii_header = nii_img.header
    nii_data = nii_img.get_fdata()
    return nii_img,nii_header,nii_data

# Getting the range of cut images.
def find_range(data1, data2, axis):
    range_list = []
    for i in range(data1.shape[axis]):
        if 1 in data1[(slice(None),)*axis + (i,)]:
            range_list.append(i)
    if len(range_list) == 0:
        start = 0
        end = data1.shape[axis] - 1
    else:
        start = range_list[0]
        end = range_list[-1]
    for i in range(data2.shape[axis]):
        if 1 in data2[(slice(None),)*axis + (i,)]:
            range_list.append(i)
    if len(range_list) == 0:
        start = 0
        end = data2.shape[axis] - 1
    else:
        start = min(start, range_list[-1])
        end = max(end, range_list[-1])
    return start - 15, end + 15

# Cutting the image.
def crop_nifti_image(data_cropped,header):
    new_header = nib.Nifti1Header()
    new_header.set_data_shape(data_cropped.shape)
    new_header.set_zooms((header['pixdim'][1],header['pixdim'][2],header['pixdim'][3]))
    new_header.set_xyzt_units(xyz='mm', t='sec')
    new_header.set_data_dtype(data_cropped.dtype)

    new_img = nib.Nifti1Image(data_cropped, None, new_header)

    return new_img

# Registration of cut images.
def rigid_registration(moving_path, fixed_path):
    
    moving_img = sitk.ReadImage(moving_path)
    fixed_img = sitk.ReadImage(fixed_path)
    
    registration = sitk.ImageRegistrationMethod()
    metric = registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    registration.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 1000)
    registration.SetInitialTransform(sitk.TranslationTransform(moving_img.GetDimension()))
    registration.SetInterpolator(sitk.sitkLinear)

    optimized_transform = registration.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32),
                                                sitk.Cast(moving_img, sitk.sitkFloat32))

    resampled = sitk.Resample(moving_img, fixed_img, optimized_transform, sitk.sitkLinear, 0.0)
    sitk.WriteImage(resampled, "nii3_regist.nii")

# Acquisition of the four-dimensional features of voxels.
def ori_featrue(ct_cut,ct_label_cut,pt_cut):
    ct_hu = []
    pet_suv = []
    ct_en = []
    pt_en = []
    voxel_index = []

    im_data_ct = ct_cut.get_fdata()
    h_ct, w_ct, d_ct = im_data_ct.shape

    im_data_label = ct_label_cut.get_fdata()
    h_label, w_label, d_label = im_data_label.shape

    im_data_pt = pt_cut.get_fdata()
    h_pt, w_pt, d_pt = im_data_pt.shape

    i = 0
    j = 0
    k = 0
    count = 0                                
    while i < h_label:
        j = 0
        k = 0
        while j < w_label:
            k = 0
            while k < d_label:           
                voxel_label = im_data_label[i,j,k]

                if voxel_label == 1 :
                    count = count+1

                    voxel_ct = im_data_ct[i,j,k]
                    voxel_ct = [voxel_ct]
                    ct_hu = ct_hu + voxel_ct

                    voxel_pt = im_data_pt[i,j,k]
                    voxel_pt = round(voxel_pt,2)
                    voxel_pt = [voxel_pt]
                    pet_suv = pet_suv + voxel_pt

                    voxel_ct_all = []
                    for l in range(-4,5):
                        for m in range(-4,5):
                            for n in range(-4,5):
                                if 0 < i+l <= h_ct and 0 < j+m <= w_ct and 0 < k+n <= d_ct:
                                    voxel_ct_entropy = im_data_ct[i+l,j+m,k+n]
                                    voxel_ct_entropy = [voxel_ct_entropy]
                                    voxel_ct_all = voxel_ct_all + voxel_ct_entropy
                                else: 
                                    voxel_ct_entropy = [0]
                                    voxel_ct_all = voxel_ct_all + voxel_ct_entropy
                    result_ct = list(pd.value_counts(voxel_ct_all))
                    ct_entropy_729 = entropy(result_ct)
                    ct_en.append(ct_entropy_729)

                    voxel_pt_all = []
                    for l in range(-4,5):
                        for m in range(-4,5):
                            for n in range(-4,5):
                                if 0 < i+l <= h_pt and 0 < j+m <= w_pt and 0 < k+n <= d_pt:
                                    voxel_pt_entropy = im_data_pt[i+l,j+m,k+n]
                                    voxel_pt_entropy = round(voxel_pt_entropy,1)
                                    voxel_pt_entropy = [voxel_pt_entropy]
                                    voxel_pt_all = voxel_pt_all + voxel_pt_entropy
                                else: 
                                    voxel_pt_entropy = [0]
                                    voxel_pt_all = voxel_pt_all + voxel_pt_entropy

                    result_pt = list(pd.value_counts(voxel_pt_all)) 
                    pt_entropy_729 = entropy(result_pt)
                    pt_en.append(pt_entropy_729)
                    a = "{:0>3d}".format(i,"000")
                    b = "{:0>3d}".format(j,"000")
                    c = "{:0>3d}".format(k,"000")

                    voxel_name = str(a)+"_"+str(b)+"_"+str(c)
                    voxel_name = voxel_name.split(' ')
                    voxel_index = voxel_index + voxel_name

                k = k + 1
            j = j + 1
        i = i + 1
    ct_hu = pd.Series(ct_hu,index = voxel_index)
    pet_suv = pd.Series(pet_suv,index = voxel_index)
    ct_en = pd.Series(ct_en,index = voxel_index)
    pt_en = pd.Series(pt_en,index = voxel_index)        
    feature_all = pd.concat([ct_hu,pet_suv,ct_en,pt_en],axis = 1)
    feature_all.columns = ['Hu','SUV','ct_entropy','pt_entropy']
    return feature_all  

# MaxMin standardization based on data from scanner 1.
def maxmin(data):
    min_hu = -539
    max_hu = 197

    min_suv = 0.18
    max_suv = 34.5

    min_ct_entropy = 2.02
    max_ct_entropy = 5.81

    min_pt_entropy = 0.51
    max_pt_entropy = 5.69
    
    data['Hu'] = (data['Hu'] - min_hu) / (max_hu - min_hu)
    data['SUV'] = (data['SUV'] - min_suv) / (max_suv - min_suv)   
    data['ct_entropy'] = (data['ct_entropy'] - min_ct_entropy) / (max_ct_entropy - min_ct_entropy)  
    data['pt_entropy'] = (data['pt_entropy'] - min_pt_entropy) / (max_pt_entropy - min_pt_entropy)

    return data

# Individual-level clustering based on the clustering centroid of scanner 1.
def k_means(df):
    a = [0.779, 0.177, 0.309, 0.672]
    b = [0.769, 0.041, 0.429, 0.382]
    c = [0.767, 0.089, 0.603, 0.655]

    a_list = []
    b_list = []
    c_list = []

    for index, row in df.iterrows():
        point = row.values

        dist_a = math.sqrt(sum([(point[i]-a[i])**2 for i in range(len(point))]))
        dist_b = math.sqrt(sum([(point[i]-b[i])**2 for i in range(len(point))]))
        dist_c = math.sqrt(sum([(point[i]-c[i])**2 for i in range(len(point))]))

        min_dist = min(dist_a, dist_b, dist_c)
        if min_dist == dist_a:
            a_list.append(index)
        elif min_dist == dist_b:
            b_list.append(index)
        else:
            c_list.append(index)

    a_df = df.loc[a_list,:]
    b_df = df.loc[b_list,:]
    c_df = df.loc[c_list,:]
    return a_df,b_df,c_df     

# Generate masks for each cluster.
def generate_masks(data1,data2,data3,image_pet):    
    image_background = image_pet.get_fdata()
    
    if not data1.empty:
        if len(list(data1.index)) >= 4:
            data1_indnames = list(data1.index)
            image_background.fill(0)
            for i in range(0,len(list(data1.index))):
                data1_n = data1_indnames[i]
                x = int(str(data1_n)[0:3])
                y = int(str(data1_n)[4:7])
                z = int(str(data1_n)[8:11])
                patient_name = str(data1_n)[12:]
                
                image_background[x, y, z] = 1
            img_mask1 = nib.Nifti1Image(image_background, image_pet.affine)
            nib.save(img_mask1,"img_mask1")
    else:
        img_mask1 = 0
               
    if not data2.empty:
        if len(list(data2.index)) >= 4:
            data2_indnames = list(data2.index)
            image_background.fill(0)
            for i in range(0,len(list(data2.index))):
                data2_n = data2_indnames[i]
                x = int(str(data2_n)[0:3])
                y = int(str(data2_n)[4:7])
                z = int(str(data2_n)[8:11])
                patient_name = str(data2_n)[12:]
                
                image_background[x, y, z] = 1
            img_mask2 = nib.Nifti1Image(image_background, image_pet.affine)
            nib.save(img_mask2,"img_mask2")
        else:
            img_mask2 = 0
    else:
        img_mask2 = 0 
               
    if not data3.empty:
        if len(list(data3.index)) >= 4:
            data3_indnames = list(data3.index)
            image_background.fill(0)
            for i in range(0,len(list(data3.index))):
                data3_n = data3_indnames[i]
                x = int(str(data3_n)[0:3])
                y = int(str(data3_n)[4:7])
                z = int(str(data3_n)[8:11])
                patient_name = str(data3_n)[12:]
                
                image_background[x, y, z] = 1
            img_mask3 = nib.Nifti1Image(image_background, image_pet.affine)
            nib.save(img_mask3,"img_mask3")            
    else:
        img_mask3 = 0
    return img_mask1,img_mask2,img_mask3

# Extract the features of the mask for each cluster.
def extraction_features(image_path,mask_path):
    para = r'Params.yaml'
    extractor = FEE.RadiomicsFeatureExtractor(para)
    extractor.enableAllImageTypes()
    result = extractor.execute(image_path,mask_path)
    results = []
    for key,value in result.items():
        couple = [key,value]
        results.append(couple)
    rad_feature = pd.DataFrame().append(results)
    return rad_feature

# Calculating ratio-habitat radiomics features.
def radio_radiomics(group1_ct,group1_pt,group2_ct,group2_pt,group3_ct,group3_pt,voxel1,voxel2,voxel3):
    radio1 = voxel1/(voxel1+voxel2+voxel3)
    radio2 = voxel2/(voxel1+voxel2+voxel3)
    radio3 = voxel3/(voxel1+voxel2+voxel3)

    def ct(x):
        return 'CT_'+ x
    
    def pt(x):
        return 'PET_'+ x
    
    def turn_feature_ct1(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:]
        min_data1 = 0
        max_data1 = 4.293        
        data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] = (data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 3.25        
        data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] = (data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] - min_data2) / (max_data2 - min_data2)
        return data
    
    def turn_feature_ct2(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:]
        min_data1 = 0
        max_data1 = 5.045               
        data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] = (data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 3.143        
        data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] = (data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] - min_data2) / (max_data2 - min_data2)
        return data

    def turn_feature_ct3(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:]
        min_data1 = 0
        max_data1 = 1.918               
        data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] = (data['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 11.578        
        data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] = (data['CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis'] - min_data2) / (max_data2 - min_data2)
        return data
    
    def turn_feature_pt1(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:]
        min_data1 = 0
        max_data1 = 3               
        data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] = (data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 0.222            
        data['PET_wavelet-HLL_glszm_ZonePercentage'] = (data['PET_wavelet-HLL_glszm_ZonePercentage'] - min_data2) / (max_data2 - min_data2)
        return data 

    def turn_feature_pt2(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:] 
        min_data1 = 0
        max_data1 = 3               
        data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] = (data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 0.6               
        data['PET_wavelet-HLL_glszm_ZonePercentage'] = (data['PET_wavelet-HLL_glszm_ZonePercentage'] - min_data2) / (max_data2 - min_data2)
        return data 

    def turn_feature_pt3(feature,form):
        title = feature.iloc[:,0]
        values = feature.iloc[:,1]
        data = pd.DataFrame(values)
        data = data.T
        data.columns = title
        data.index = ["feature"]
        data.rename(columns=form, inplace = True)
        data = data.iloc[:,22:]
        min_data1 = 0
        max_data1 = 3.1               
        data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] = (data['PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis'] - min_data1) / (max_data1 - min_data1)
        min_data2 = 0
        max_data2 = 0.007               
        data['PET_wavelet-HLL_glszm_ZonePercentage'] = (data['PET_wavelet-HLL_glszm_ZonePercentage'] - min_data2) / (max_data2 - min_data2)
        return data 

    group1_ct = turn_feature_ct1(group1_ct,ct).applymap(lambda x: x * float(radio1))
    group1_pt = turn_feature_pt1(group1_pt,pt).applymap(lambda x: x * float(radio1))
    group2_ct = turn_feature_ct2(group2_ct,ct).applymap(lambda x: x * float(radio2))
    group2_pt = turn_feature_pt2(group2_pt,pt).applymap(lambda x: x * float(radio2))
    group3_ct = turn_feature_ct3(group3_ct,ct).applymap(lambda x: x * float(radio3))
    group3_pt = turn_feature_pt3(group3_pt,pt).applymap(lambda x: x * float(radio3))
    df_ct = group1_ct.add(group2_ct, fill_value=0).add(group3_ct, fill_value=0)
    df_pt = group1_pt.add(group2_pt, fill_value=0).add(group3_pt, fill_value=0)
    df = pd.concat([df_ct,df_pt],axis=1)
    feature_list = ['CT_wavelet-HLL_glrlm_LongRunLowGrayLevelEmphasis', 'PET_wavelet-HLL_glszm_HighGrayLevelZoneEmphasis', 
                    'PET_wavelet-HLL_glszm_ZonePercentage', 'CT_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis']
    df = df.loc[:,feature_list]
    return df

# Calculating predictive probabilities.
def prediction(df_train,X_test): 
    X_train = df_train.drop(['label'],axis=1)
    y_train = df_train[['label']]
    X_test = X_test
    model = xgb.XGBClassifier(n_estimators=50,max_depth=5,objective='binary:logistic',use_label_encoder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    negative_prob = y_pred[0][0]
    positive_prob = y_pred[0][1]
        
    return negative_prob,positive_prob


# In[3]:


class NiiViewer:
    # Creating a Visualization Window.
    def __init__(self, master):
        self.master = master
        self.master.title("Prediction of efficacy of NAC for breast cancer")
        self.master.geometry("800x320")
        self.master.configure(bg="lightgrey")
        
        self.path1_entry = tk.Entry(self.master, width=50)
        self.path1_entry.grid(row=0, column=0, padx=10, pady=10)
        self.select1_button = tk.Button(self.master, text="CT image path", command=self.select_file1)
        self.select1_button.grid(row=0, column=1, padx=10, pady=10)


        self.path2_entry = tk.Entry(self.master, width=50)
        self.path2_entry.grid(row=1, column=0, padx=10, pady=10)
        self.select2_button = tk.Button(self.master, text="CT mask path", command=self.select_file2)
        self.select2_button.grid(row=1, column=1, padx=10, pady=10)

        self.path3_entry = tk.Entry(self.master, width=50)
        self.path3_entry.grid(row=2, column=0, padx=10, pady=10)
        self.select3_button = tk.Button(self.master, text="PET image path", command=self.select_file3)
        self.select3_button.grid(row=2, column=1, padx=10, pady=10)
        
        self.path4_entry = tk.Entry(self.master, width=50)
        self.path4_entry.grid(row=3, column=0, padx=10, pady=10)
        self.select4_button = tk.Button(self.master, text="PET mask path", command=self.select_file4)
        self.select4_button.grid(row=3, column=1, padx=10, pady=10)
        
        self.path5_entry = tk.Entry(self.master, width=10)
        self.path5_entry.grid(row=4, column=0, padx=10, pady=10)
        self.path5_label = tk.Label(self.master, text="Aortic arch SUVmean", fg='black')
        self.path5_label.grid(row=4, column=1, sticky='E', padx=10, pady=10)
        
        self.show_button = tk.Button(self.master, text="Start prediction", command=self.show_file_info)
        self.show_button.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

        self.output_box = tk.Text(self.master, width=30, height=10)
        self.output_box.grid(row=0, column=2, rowspan=4, padx=10, pady=10)

    def write(self, msg):
        self.output_box.insert(tk.END, msg)
        
    def select_file1(self):
        path1 = filedialog.askopenfilename(filetypes=[("Nii Files", "*.nii")])
        self.path1_entry.delete(0, tk.END)
        self.path1_entry.insert(0, path1)

    def select_file2(self):
        path2 = filedialog.askopenfilename(filetypes=[("Nii Files", "*.nii")])
        self.path2_entry.delete(0, tk.END)
        self.path2_entry.insert(0, path2)
        
    def select_file3(self):
        path3 = filedialog.askopenfilename(filetypes=[("Nii Files", "*.nii")])
        self.path3_entry.delete(0, tk.END)
        self.path3_entry.insert(0, path3)
        
    def select_file4(self):
        path4 = filedialog.askopenfilename(filetypes=[("Nii Files", "*.nii")])
        self.path4_entry.delete(0, tk.END)
        self.path4_entry.insert(0, path4)
        
    def show_file_info(self):
        nii1_path = self.path1_entry.get()
        nii1_img,nii1_header,nii1_data = load_nii_file(nii1_path)

        nii2_path = self.path2_entry.get()
        nii2_img,nii2_header,nii2_data = load_nii_file(nii2_path)

        nii3_path = self.path3_entry.get()
        nii3_img,nii3_header,nii3_data = load_nii_file(nii3_path)

        nii4_path = self.path4_entry.get()
        nii4_img,nii4_header,nii4_data = load_nii_file(nii4_path)

        x_start, x_end = find_range(nii2_data, nii4_data, 0)
        y_start, y_end = find_range(nii2_data, nii4_data, 1)
        z_start, z_end = find_range(nii2_data, nii4_data, 2)
        
        data_cropped_1 = nii1_data[x_start:x_end, y_start:y_end, z_start:z_end]
        nii1_cut = crop_nifti_image(data_cropped_1,nii1_header)
        nib.save(nii1_cut, "nii1_cut")
        data_cropped_2 = nii2_data[x_start:x_end, y_start:y_end, z_start:z_end]
        nii2_cut = crop_nifti_image(data_cropped_2,nii2_header)
        data_cropped_3 = nii3_data[x_start:x_end, y_start:y_end, z_start:z_end]
        nii3_cut = crop_nifti_image(data_cropped_3,nii3_header)
        nib.save(nii3_cut, "nii3_cut")
        
        nii3_regist = rigid_registration("nii3_cut","nii1_cut")
        nii3_regist = nib.load("nii3_regist.nii")
        feature_all = ori_featrue(nii1_cut,nii2_cut,nii3_regist)
        suv_mean = float(self.path5_entry.get())
        feature_all['SUV'] = feature_all['SUV']/suv_mean
        
        data_maxmin = maxmin(feature_all)
        
        group1,group2,group3 = k_means(data_maxmin)
        mask1,mask2,mask3 = generate_masks(group1,group2,group3,nii3_regist)
        
        if mask1 != 0:
            mask_nii_1 = nib.load('img_mask1.nii')
            mask_data1 = mask_nii_1.get_fdata()
            num_voxels_1 = np.count_nonzero(mask_data1 == 1)

            ct_group1_feature = extraction_features("nii1_cut.nii","img_mask1.nii")            
            pt_group1_feature = extraction_features("nii3_cut.nii","img_mask1.nii")
        else:
            ct_group1_feature = pd.read_csv("empty.csv",index_col = 0)
            pt_group1_feature = pd.read_csv("empty.csv",index_col = 0)
            num_voxels_1 = 0
            
        if mask2 != 0:
            mask_nii_2 = nib.load('img_mask2.nii')
            mask_data2 = mask_nii_2.get_fdata()
            num_voxels_2 = np.count_nonzero(mask_data2 == 1)

            ct_group2_feature = extraction_features("nii1_cut.nii","img_mask2.nii")
            pt_group2_feature = extraction_features("nii3_cut.nii","img_mask2.nii") 
        else:
            ct_group2_feature = pd.read_csv("empty.csv",index_col = 0)
            pt_group2_feature = pd.read_csv("empty.csv",index_col = 0)
            num_voxels_2 = 0
            
        if mask3 != 0:
            mask_nii_3 = nib.load('img_mask3.nii')
            mask_data3 = mask_nii_3.get_fdata()
            num_voxels_3 = np.count_nonzero(mask_data3 == 1)
            
            ct_group3_feature = extraction_features("nii1_cut.nii","img_mask3.nii")
            pt_group3_feature = extraction_features("nii3_cut.nii","img_mask3.nii")
        else:
            ct_group3_feature = pd.read_csv("empty.csv",index_col = 0)
            pt_group3_feature = pd.read_csv("empty.csv",index_col = 0)
            num_voxels_3 = 0
            
        radio_radiomics_feature = radio_radiomics(ct_group1_feature,pt_group1_feature,
                                                  ct_group2_feature,pt_group2_feature,
                                                  ct_group3_feature,pt_group3_feature,
                                                  num_voxels_1,num_voxels_2,num_voxels_3)
        train_data = pd.read_csv(r"model_data.csv",index_col = 0)
        a,b = prediction(train_data,radio_radiomics_feature)
        viewer.write(f"The probability of predicting pCR is: {round(a, 3):.3f}\n")
        
if __name__ == "__main__":
    root = tk.Tk()
    viewer = NiiViewer(root)
    root.mainloop()


# In[ ]:




