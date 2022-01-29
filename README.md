# LEGO_sorter
Python code to run with an EPSON SCARA roboy work cell for LEGO part sorting

# How to run
Clone this repository

Pip install the requirements
NOTE: $pip install torch installs different version depending on the used platform,
we recommend installing torch version that supports CUDA 11.3 or newer. To do this we 
recommend visiting https://pytorch.org/ and determining right pip install command
following instructions stated there.

Run the Communicate_to_robot.py script

After the script prints out:

"#########################"
     "Starting server"    
"#########################"

then start the robot.

