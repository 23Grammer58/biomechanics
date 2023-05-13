********************************************************

         Galaxy python installation for Windows

                       2019-11-20

********************************************************


Python2.7(3.5) gxipy installation  
=================================

1.Install python2.7(3.5)

  (1) Download the python2.7(3.5) installation package for Windows(x86/x86_64) from the python official path as follows and perform installations.

      Download path: https://www.python.org/downloads/windows/
  
  (2) Add the path of python.exe to the system environment variable path.

2.Install pip tools

  (1) Enter https://pip.pypa.io/en/stable/installing/ in browser and download the get-pip.py.

  (2) Bring up the DOS command window by typing CMD, and switch to the path containing get-pip.py.

  (3) Type the command as follows in DOS command window to install pip.
    
      python get-pip.py

  (4) Add the path of pip.exe to the system environment variable path.

3.Install numpy library
 
  Type the command as follows in DOS command window.

      pip install numpy

Attention
=================================
   (1) The sample may depend on third party libraries(e.g. PIL), please install it by yourself.

   (2) Python samples can only be executed if they are placed in the same directory as gxipy.

   (3) If the program developed by users depends on the gxipy library, gxipy in current directory needs to be copied to the users development directory.
  



