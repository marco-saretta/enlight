# ENLIGHT Documentation

Source files for the MkDocs site live in `docs/source/`.

## Build locally

```bash
uv run --with mkdocs-material --with mkdocstrings-python mkdocs serve -f docs/mkdocs.yaml
```

## Deploy

Pushed automatically to GitHub Pages on every merge to `main`
via `.github/workflows/docs.yml`.
