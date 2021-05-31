# Collision_Detection

The goal of this program is detect obstacles using only the monocular camera on the tello drone.

The implementation is largely based around this paper: https://staff.fnwi.uva.nl/a.visser/education/masterProjects/Obstacle%20Avoidance%20using%20Monocular%20Vision%20on%20Micro%20Aerial%20Vehicles_final.pdf

# Dependencies

Python 2.7

OpenCV - Installation instructions for linux here: https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html

scikit-learn - Installation instructions for linux here: https://scikit-learn.org/stable/install.html

# Installation

First make sure you have pip for Python 2.7 installed. Do this with the following commands:

    curl -O https://bootstrap.pypa.io/pip/2.7/get-pip.py
    python get-pip.py
    python -m pip install --upgrade "pip < 21.0"

Running `pip -V` should return pip 20.x.x

Now clone the repo and install dependencies using:

    git clone https://github.com/GDP-Drone-2021/Collision_Detection.git
    cd Collision_Detection
    pip install -r requirements.txt

You should now be able to run the program using:

    python CollisionDetector.py
