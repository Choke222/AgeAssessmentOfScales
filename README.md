# Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing

This repository is accompanying the papers "Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing" (G. Suzuki, M. Nishiyama, R. Hoson, K. Yoshida, H. Shioya, and K. Shimoda, IEEE ACCESS 2024, ([doi:10.1109/ACCESS.2024.3396818](https://ieeexplore.ieee.org/document/10520289?source=authoralert))).

## File List
The following files are provided in this repository:
- src/all.sh: Age assessment programs 
- dataset/: Uroko images
- src/predict.py: Only resting zone detection
- src/main.py: Only age assessment 

## Usage
Environment
- Docker version 25.0.0, build e758fe5

### Build a Docker environment
Compose up the Docker container and log in to the container.
```bash
docker-compose up -d
docker-compose exec age_estimate_public bash
docker-container:/workspace# cd estimation_src
docker-container:/workspace/estimation_src# sh first.sh
```
Download the model weights from [URL](https://drive.google.com/file/d/1gAy2jpc6JLyAerJsBqkpzHcF1jvm81W2/view?usp=sharing) and store them in the following directory
docker-container:/workspace/estimation_src/runs/models/pspnet_vgg16/uroko_w/

### Quick Test for Age Assessment
The age assessment is performed by executing the following shell file, specifying the first line of all.sh as the name of the image file in dataset/image.
```bash
docker-container:/workspace/estimation_src# sh all.sh
```

### Inference: Resting zone detection
The following code can be used to split an image into patches.
```bash
docker-container:/workspace/estimation_src# python3 data_proc.py ../dataset/image/uroko20181a4.jpg
```

You can perform resting zone detection by executing the following code.
```bash
docker-container:/workspace/estimation_src# python3 predict.py ../dataset/patching/uroko20181a4 ../dataset/out_restingzonedetec
```

Merging patches of images can be performed with the following code.
```bash
docker-container:/workspace/estimation_src# python3 image_connect.py ../dataset/out_restingzonedetec/uroko20181a4 ../dataset/restzone/uroko20181a
```

### Inference: Age assessment
You can perform resting zone detection by executing the following code.
```bash
docker-container:/workspace/estimation_src# python3 main.py -r uroko20181a4 -d uroko20181a4
```

## Acknowledgements
This research was supported by the Strategic Information and Communications R&D Promotion Programme (SCOPE, 16771288), Ministry of Internal Affairs and Communications.

## License and Referencing
This program is licensed under the MIT License. If you in any way use this code for research that results in publications, please cite our original article listed above.
You can use the following BibTeX entry.
```bibtex
@article{10520289,
  author={Suzuki, Genki and Nishiyama, Mikiyasu and Hoson, Ryoma and Yoshida, Katsunobu and Shioya, Hiroyuki and Shimoda, Kazutaka},
  journal={IEEE Access}, 
  title={Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing}, 
  year={2024},
  volume={12},
  number={},
  pages={64779-64794},
  doi={10.1109/ACCESS.2024.3396818}}
```

[The MIT License (MIT) | Open Source Initiative](https://opensource.org/license/mit)
