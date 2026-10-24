# One-time: copy data/ into the ff-assistant_ffa-data Docker volume, then start the stack
# on it. data/ on Windows is left untouched as a backup (it stops updating, except
# data/reports, which stays shared). Run from the repo folder, after `git pull`.
$ErrorActionPreference = "Stop"

docker compose down
$env:FFA_GIT_SHA = git rev-parse --short HEAD
docker compose build
docker compose up --no-start          # creates the empty ff-assistant_ffa-data volume

Write-Host "Copying data/ into the volume (a few minutes)..."
docker run --rm -v ff-assistant_ffa-data:/to -v "${PWD}/data:/from:ro" alpine `
  sh -c "cp -a /from/. /to/ && echo files: && find /to -name '*.parquet' | wc -l"

docker compose up -d
Write-Host "Done. Check: docker compose ps"
