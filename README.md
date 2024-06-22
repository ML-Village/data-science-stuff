# Data Science Stuff

## Dockerfiles

Dockerfiles are prepared to spin up required games locally. Refer below for scripts on how to build and run these games in docker.

### Pokemon Showdown

```bash
# remove any existing image
docker rmi pokemon-showdown

# build pokemon-showdown docker image
docker build -t pokemon-showdown -f dockerfiles/pokemon-showdown.Dockerfile .

# run pokemon-showdown game at localhost:8000
docker run --rm -p 8000:8000 pokemon-showdown
```
