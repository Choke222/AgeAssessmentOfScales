# Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing

This repository is accompanying the papers "Age Assessment Using Chum Salmon Scale by Neural Networks and Image Processing" (G. Suzuki, M. Nishiyama, R. Hoson, K. Yoshida, H. Shioya, and K. Shimoda, IEEE ACCESS 2024, [doi:xxx]([https:xxxxx](https://github.com/Choke222/AgeAssessmentOfScales)))).

The papers can be found at XXX below.

## File List
The following files are provided in this repository:
- predict.py: Resting zone detection
- main.py: Age assessment
- dataset/

## Usage

Environment
- Docker version 25.0.0, build e758fe5

### Build a Docker environment
Compose up the Docker container and log in to the container.
```bash
docker-compose up -d
docker-compose exec age_estimate bash
```

### Inference: Resting zone detection
You can perform resting zone detection by executing the following code.
```bash
docker-container:/workspace/estimation_src# python3 predict.py 
```

### Inference: Age assessment
You can perform resting zone detection by executing the following code.
```bash
docker-container:/workspace/estimation_src# python3 main.py -r test -d test
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

The MIT License (MIT) | Open Source Initiative
