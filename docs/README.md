# HPVsim docs (Quarto)

Published documentation lives at [https://docs.hpvsim.org](https://docs.hpvsim.org).

## Build locally

1. Install [Quarto](https://quarto.org/docs/get-started/) and Python dependencies:

   ```bash
   pip install -e ..
   pip install -r requirements.txt
   quarto add machow/quartodoc --no-prompt
   hpvsim-download-data
   ```

2. From this directory, render the site:

   ```bash
   quarto render
   ```

   Output is written to `_site/`. The `pre-render` step runs `quarto_utils.py` to refresh API pages (`quartodoc`) and `post-render` cleans temporary tutorial artifacts.

3. To preview:

   ```bash
   quarto preview
   ```

Notebooks under `tutorials/` are executed during render when Quarto’s execute settings allow it; ensure data are downloaded first.
