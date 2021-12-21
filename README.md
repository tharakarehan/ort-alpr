# ORT-ALPR (ONNXRunTime Automatic License Plate Recognition)

<div align="justify">This is an ALPR framework (Automatic License Plate Recognition) that is optimised for cpu inferencing on Onnxruntime. This framework is capable of detecting vehicle license plates, recognizing license plates and blurring license plates. For detecting license plates pre-trained <a href="http://sergiomsilva.com/pubs/alpr-unconstrained/" target="_blank">WPOD-NET</a> has been used. For OCR (Optical Character Recognition), <a href="https://github.com/PaddlePaddle/PaddleOCR" target="_blank"> paddle's OCR</a> model has been used. All the models are converted to  ONNX for inferencing on Onnxruntime</div>

<p align="center">
  <img src="https://github.com/tharakarehan/ort-alpr/blob/master/collage.jpeg">
</p>

## Installation

Create a new conda environment. If you dont have conda installed download [miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create -n ort-alpr python=3.8
```
Clone this repository to your computer and navigate to the directory.

Activate new enviroment
```bash
conda activate ort-alpr  
```
Install all the libraries used
```bash
pip install -r requirements.txt  
```

## Usage

Initialize the ALPR class

```bash
x = ALPR(out_dir='lpd_results')
```

### Detecting Mode

```bash
x.detect_lp(path ='Test_Images/Cars438.jpeg',Bbox=False,show=True,save=False)
```

### Recognizing Mode

```bash
x.recognize_lp(path ='Test_Images/Cars450.jpeg',show=True,save=False,f_scale=1.5)
```

### Blurring Mode

```bash
x.blur_lp(path ='Test_Images/Cars422.png',show=True,save=False)
```

<p align="center">
  <img src="https://github.com/tharakarehan/ort-alpr/blob/master/blur-collage.png">
</p>

## License
[MIT](https://choosealicense.com/licenses/mit/)


