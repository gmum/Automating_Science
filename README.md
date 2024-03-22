# Automating Science

The repository for the course "Automating Science" taught at the Jagiellonian University in 2024. The page is hosted by the machine learning research group GMUM.

## Theme

How Machine Learning can facilitate scientific discoveries. We will discuss how we make progress and potential role of applying AI methods into the scientific problems.

## Schedule

- lectures: Wednesdays 4pm
- tutorials: Fridays 10.15am

We highly encourage you to attend the classes and engage with your colleagues.

## Lecture content

- [Pegaz](https://pegaz.uj.edu.pl/course/view.php?id=625462)
- [Notes from the lectures](https://docs.google.com/presentation/d/1IOJG0ixTsbYHxkcp_Vo_vCG90n5zfwVJipOlrzPrd7Q/edit)
- [Book](https://fleuret.org/public/lbdl.pdf)

## Grading

Exam 25% at the end of semester

Project 50% composed of:

- first stage 20 %
- mid stage 20 %
- final stage 60 %

Homework 15% where:

- top 2 notebooks 50%
- rest notebooks 50%

Piazza / lecture / tutorial activity 10%

## Piazza

[Piazza Automating Science 2024](https://piazza.com/uj.edu.pl/spring2024/wmiiiasudlnm)

## Contact

If possible, by Piazza. If not successful, then reach us by emails.

## Set up

data directory (added to gitignore) will store the data: mnist, fashionmnist, cifar, synbols
models directory (added to gitignore) will store the pretrained / results of trained models
jtt directory for just-train-twice (week 2)
all_about_uncertainty (weeks 3,4)

Install conda (follow [the official instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) )

```bash
conda env create -f environment.yml
```

## Info

(please do not complete the notebooks before the tutorials)

- [x] Week 1: introduction

  - intro
  - python: numpy, torch
  - first ML trainings

- [x] Week 2: shortcut learning
- [x] Week 3: uncertainty
- [x] Week 4: uncertainty and anomaly detection
- [ ] Week 5: [remote] Project discussions
- [ ] Week 6: transformers
- [ ] Week 7: continual / multi learning
- [ ] Week 8: drug discovery
- [ ] Week 9+: into the wild / project meetings
