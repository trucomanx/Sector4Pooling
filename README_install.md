# Requirements
Install the requirements.

    pip3 install -r requirements.txt

# Packaging

Download the source code

    git clone https://github.com/trucomanx/Sector4Pooling

The next command generates the `dist/Sector4Pooling-VERSION.tar.gz` file.

    cd Sector4Pooling/src
    python setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/Sector4Pooling-VERSION.tar.gz
