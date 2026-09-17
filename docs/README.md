# ENLIGHT Documentation

Source files for the MkDocs site live in `docs/source/`.

## Build locally

```bash
uv run --with "mkdocs<2" --with mkdocs-material --with mkdocstrings-python mkdocs serve -f docs/mkdocs.yaml
```

## Deploy

Not automated yet. To publish to GitHub Pages manually:

```bash
uv run --with "mkdocs<2" --with mkdocs-material --with mkdocstrings-python mkdocs gh-deploy -f docs/mkdocs.yaml
```
