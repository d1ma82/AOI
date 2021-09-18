

compile:
    release
cl.exe /W4 /Zi /EHsc main.cpp ./utils/multiple_window.cpp opencv_core453.lib opencv_highgui453.lib opencv_imgproc453.lib opencv_imgcodecs453.lib

launch sample:
    ./main.exe "../source/test.pgm" ''  -removelight=0 -lightMethod=1