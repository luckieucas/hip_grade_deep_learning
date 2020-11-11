from glob import glob 
img_list = glob("./sample_images/onfh_3cls/sample/*png")
label_dict={'normal':0,'oa_I':1,'oa_II':2,'oa_III':3,'onfh_II':4,'onfh_III':5,'onfh_IV':6}
with open("./sample_images/onfh_3cls/test.txt",'w') as f:
    for img in img_list:
        label_list =['0']*3
        img_name = img.split("/")[-1]
        print(img_name)
        #label = label_dict[img_name.replace("_R_","_L_").split("_L_")[-1].replace(".png","")] #hip 7cls
        label = img_name.split("_")[-2]
        label_list[int(label)] = '1'
        label_str = " ".join(label_list)
        print(label)
        f.write(img_name+' '+label_str + '\n')
print(len(img_list))