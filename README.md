# Parts of Speech tagging using a Hidden Markov Model (HMM)

--------

This repository is the result of a project we did as part of our computational syntax course, which is part of the syllabus of the Master in Language Analysis and Processing on EHU/UPV.

--------

## Objectives

- Implement a Hidden Model for PoS tagging from scratch
- Experiment on different languages  from the universal dependencies databank

## What is POS tagging with HMMs? 

Placeholder text

## Datasets

Placeholder text

- Czech
- Slovak
- English
- Spanish
- Persian
- German
- Urdu
- Gilaki

## Try out the models!

Docker compose can be used in this project to automatically setup the
marimo server. Just use: 

```sh
docker-compose up --build
```

and enter the address indicated into the browser, replacing `0.0.0.0` for `localhost`, 
or `127.0.0.1`.

If your linux distribution has not added the user to the `docker`
group, you might need to use root permissions in order to run
`docker-compose` (`sudo docker-compose` or `doas docker-compose`).


