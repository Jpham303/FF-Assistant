# Run the ffa CLI inside the stack (the data lives in a Docker volume), e.g.:
#   .\scripts\ffa.ps1 report --week 7
docker compose exec dagster-daemon ffa @args
