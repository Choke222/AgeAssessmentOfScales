# Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing

This repository is accompanying the papers "Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing" (G. Suzuki, M. Nishiyama, R. Hoson, K. Yoshida, H. Shioya, and K. Shimoda, IEEE ACCESS 2024, [doi:xxx]([https:xxxxx](https://github.com/Choke222/AgeAssessmentOfScales)))).

The papers can be found at XXX below.

## File List
The following files are provided in this repository:
- predict.py: Resting zone detection
- main.py: Age assessment
- dataset/: Uroko images

## Usage

Environment
- Docker version 25.0.0, build e758fe5

### Build a Docker environment
Compose up the Docker container and log in to the container.
```bash
docker-compose up -d
docker-compose exec age_estimate_public bash
```
Download the model weights from [URL](https://drive.google.com/file/d/1gAy2jpc6JLyAerJsBqkpzHcF1jvm81W2/view?usp=sharing) and store them in the following directory
estimation_src/runs/models/pspnet_vgg16/uroko_w/

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
docker-container:/workspace/estimation_src# python3 predict.py ../dataset/patchimg/uroko20181a4 ../dataset/out_restingzonedetec
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
@article{Suzuki2024AgeAssement,
  title = {Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing},
  author = {Suzuki Genki, Nishiyama Mikiyasu, Hoson Ryoma, Yoshida Katsunobu, Shioya Hiroyuki and Shimoda Kazutaka},
  journal = {IEEE Access},
  year = {2024},
  volume = {xx},
  number = {xx},
  pages = {xxxx--xxxx},
  doi = {xx}
}
```

[The MIT License (MIT) | Open Source Initiative](https://opensource.org/license/mit)
