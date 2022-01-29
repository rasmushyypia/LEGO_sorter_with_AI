## Add CVAT-tool created Yolo formatted object train data files here

This folder should contain annotation text files corresponding to the class names that will be trained to the Yolov5 detector. These text files will be used to extract the LEGO part templates from the provided camera images. Each annotation text file should include bounding box information to the corresponding LEGO part image from where the templeta will be extracted. The file names should match (2_b.png and 2_b.txt). The text files can be generated with the CVAT-tool. More about CVAT tool at: https://github.com/openvinotoolkit/cvat

The object training data in the files should be in this order: 

label x-centre y-centre width height

0 0.496371 0.472620 0.142586 0.068302

