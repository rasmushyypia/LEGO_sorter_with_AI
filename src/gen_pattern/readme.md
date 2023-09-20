# Generation for calibration patterns

See additional information for the use from https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html


# Calibration targets for application

``
python gen_pattern.py -o c:\temp\radon_checkerboard.svg --rows 15 --columns 17 --type radon_checkerboard --square_size 10 -m 1 1 1 2 2 1
``

``
python gen_pattern.py -o c:\temp\circleboard.svg --rows 10 --columns 10 --type circles --square_size 3 --radius_rate 3

python gen_pattern.py -o c:\temp\circleboard_A4.svg -a A4 --rows 14 --columns 10 --type circles --square_size 20 --radius_rate 5

python gen_pattern.py -o c:\temp\circleboard_A3.svg -a A3 --rows 14 --columns 10 --type circles --square_size 30 --radius_rate 5

``

``
python gen_pattern.py -o c:\temp\radon_checkerboard_30.svg -a A3 --rows 14 --columns 9 --type radon_checkerboard --square_size 30 -m 1 1 1 2 2 1

python gen_pattern.py -o c:\temp\radon_checkerboard_30.svg -a A3 --rows 14 --columns 9 --type radon_checkerboard --square_size 30 -m 3 6 4 6 4 7

python gen_pattern.py -o c:\temp\radon_checkerboard_40.svg -a A3 --rows 10 --columns 7 --type radon_checkerboard --square_size 40 -m 3 4 4 4 4 5
``

``
python gen_pattern.py -o c:\temp\checkerboard_30.svg -a A3 --rows 14 --columns 9 --type checkerboard --square_size 30

python gen_pattern.py -o c:\temp\checkerboard_40.svg -a A3 --rows 10 --columns 7 --type checkerboard --square_size 40

``
