# DeepLabCut Tools for XROMM
Integrate [XMALab](https://bitbucket.org/xromm/xmalab) and [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) for high-throughput XROMM. Pipeline created by J.D. Laurence-Chasen.

See (and cite) our [methods paper](https://jeb.biologists.org/content/early/2020/07/13/jeb.226720) in the Journal of Experimental Biology.

The default workflow involves training a single DeepLabCut network that analyzes videos from both camera planes. If you want separate networks for each camera plane, use the two-network notebook in `templates/`.

Special thanks to Ben Knorlein for XMALab and Mackenzie Mathis, Alexander Mathis, Tanmay Nath, and the rest of the DeepLabCut contributors.

## Recommended path for most users
Most scientists should start with the notebook workflow, using an existing DeepLabCut environment that can already run Jupyter notebooks.

1. Install XMALab and confirm that your DeepLabCut environment is working.
2. Clone or download this repository.
3. Open a terminal in the cloned `XROMM_DLCTools` folder and start Jupyter Notebook or JupyterLab from your DeepLabCut environment there.
4. Open one of the example notebooks in `templates/`:
   - `templates/XROMM_Pipeline_Demo.ipynb`
   - `templates/XROMM_Pipeline_Demo_2Networks.ipynb`
5. Follow the notebook instructions to organize your trial folders and run the workflow.

If you already have a working XMALab + DeepLabCut setup, this is usually the easiest way to use the repo. Starting Jupyter from the cloned `XROMM_DLCTools` folder ensures the notebooks can import the local `xrommtools` package directly from this repository.

## Optional `uv`-managed setup
This repository also supports a packaged `uv` workflow for users who want a project-managed local Python environment and command-line interface.

Install `uv` by following the instructions at <https://docs.astral.sh/uv/getting-started/installation/>.

For lightweight work that does not require DeepLabCut imports:

```bash
uv lock
uv sync --no-group dlc
```

For DeepLabCut-dependent commands:

```bash
uv sync --group dlc
```

## Workflow options
### Notebook walkthroughs
The primary end-to-end examples live in:
- `templates/XROMM_Pipeline_Demo.ipynb`
- `templates/XROMM_Pipeline_Demo_2Networks.ipynb`

### Command-line interface
Inspect the packaged command surface:

```bash
uv run python -m xrommtools --help
```

The packaged CLI includes commands for:
- XMALab-to-DLC ingestion
- DLC-to-XMALab export
- video prediction orchestration
- corrected-frame augmentation
- temporal optimization review reports
- stereo triangulation review reports

The command-line workflow is intended for advanced users. Most users should start with the notebooks.

## Data layout expectations
Each trial folder should contain:
- one XMALab-exported distorted 2D points CSV
- camera media for both views, either as AVI files or camera image folders

Example:
- `trainingdata/`
  - `trial01/`
    - `trial01_2dpts.csv`
    - `trial01_camera1.avi`
    - `trial01_camera2.avi`
  - `trial02/`
    - `trial02_2dpts.csv`
    - `trial02_camera1.avi`
    - `trial02_camera2.avi`

The example fixture layout under `templates/trainingdata/` matches the expected ingestion structure.

## Development and validation
Developer setup, regression harness usage, and cross-repo validation notes are documented in `CONTRIBUTING.md`.
