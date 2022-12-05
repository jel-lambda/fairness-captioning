import os

base = "./" # directory which have backgrounds folder and final folder
bg_folder = os.path.join(base, 'backgrounds/')
final_folder = os.path.join(base, 'final/')

bg_classes = os.listdir(bg_folder)


for bg_class in bg_classes:
    bg_images = os.listdir(os.path.join(bg_folder, bg_class))
    i = 100
    for image in bg_images:

        if not image.endswith('.jpg'):
            pass
        #print(bg_class)
        print(os.path.join(bg_folder, bg_class, image))
        print(bg_folder+bg_class+'/_'+str(i)+'.jpg')
        print(os.path.join(final_folder,image.split(".")[0]))
        print(final_folder+bg_class+"_"+str(i))
        os.rename(os.path.join(bg_folder, bg_class, image), bg_folder+bg_class+'/'+bg_class+"_"+str(i)+'.jpg')
        os.rename(os.path.join(final_folder,image.split(".")[0]),final_folder+bg_class+"_"+str(i))
        i += 1

for bg_class in bg_classes:
    bg_images = os.listdir(os.path.join(bg_folder, bg_class))
    i = 0
    for image in bg_images:
        if not image.endswith('.jpg'):
            pass
        #print(bg_class)
        os.rename(os.path.join(bg_folder, bg_class, image), bg_folder+bg_class+'/'+bg_class+"_"+str(i)+'.jpg')
        os.rename(os.path.join(final_folder,image.split(".")[0]),final_folder+bg_class+"_"+str(i))
        i += 1