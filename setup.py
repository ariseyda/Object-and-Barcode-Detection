from cx_Freeze import setup, Executable



setup(name="Object_Barcode_Software",
      version="0.1",
      description="Object Detection and Barcode Reading",
      executables=[Executable("Object_and_Barcode_Detection.py")])
