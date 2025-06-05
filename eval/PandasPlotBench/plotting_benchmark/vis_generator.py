import json
import subprocess
from pathlib import Path, PurePosixPath

import nbformat as nbf
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import nbformat
import re
import os

def remove_ansi_escape(text: str) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)

def remove_bytes_literal(text: str, min_length=100) -> str:
    """
    Replace long bytes literals (b'...') in the error string with 'b`...`',
    but preserve short ones.
    """
    def replace_if_long(m):
        content = m.group(0)
        stripped = content[2:-1] if content[1] in "\"'" else content
        if len(stripped) >= min_length:
            return "b`...`"
        else:
            return content 

    pattern = r"b(['\"])(?:\\.|[^\\])*?\1"
    return re.sub(pattern, replace_if_long, text, flags=re.DOTALL)

def truncate_error_message(error: str) -> str:
    head = 1000
    tail = 1000
    if len(error) <= head + tail:
        return error
    return error[:head] + "\n...\n".format(len(error) - head - tail) + error[-tail:]


def read_jsonl(file_path: str | Path) -> list[dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path: str | Path) -> None:
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def read_responses(
    responses_file: str | Path | None = None, responses: list[dict] | None = None
) -> dict:
    if responses_file is None and responses is None:
        raise ValueError("Either response_file or responses must be provided.")

    if responses_file is not None and responses is not None:
        print(
            "Both responses file and responses list provided. Responses list would be used."
        )

    if responses is None:
        responses = read_jsonl(responses_file)

    responses_dict = dict()

    for entry in responses:
        if "id" in entry:
            responses_dict[entry["id"]] = entry

    return responses_dict


def add_index_to_filename(
    folder: str, filename: str, postfix: str = ""
) -> tuple[Path, Path | None]:
    results_file_base = Path(folder) / filename
    results_file_base = results_file_base.with_stem(results_file_base.stem + postfix)
    results_file = results_file_base.with_stem(results_file_base.stem + "_0")

    i = 0
    last_existing_file = None
    while results_file.exists():
        last_existing_file = results_file
        i += 1
        results_file = results_file_base.with_stem(results_file_base.stem + f"_{i}")

    return results_file, last_existing_file


class VisGenerator:
    """
    Object that runs generated code to build plots.
    At init pass:

    dataset: dataset
    output_file: output file to save results.
    The output is ammendment of the LLM responses logs. So, you can pass same path.
    temp_dir: dir for notebook used for plotting plots.
    """

    def __init__(
        self,
        output_folder: str | Path,
        dataset: Dataset,
        csv_folder: str | Path,
        config: DictConfig | None = None,
    ) -> None:
        self.output_folder = Path(output_folder)
        self.plots_nb_path = self.output_folder / "all_plots.ipynb"
        self.config = config
        self.csv_folder = Path(csv_folder)
        self.check_csv(dataset)
        if self.config.run_mode == "self_debug":
            self.debug_plots_nb_path = self.config.debug.output_dir

    def check_csv(self, dataset: Dataset) -> None:
        for item in dataset:
            csv_path = self.csv_folder / f"data-{item['id']}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Unpacked csv datafile not found on {csv_path}"
                )

    def build_new_nb(self, blocks: list) -> None:
        """
        save codeblocks into notebook
        """

        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_code_cell(block) for block in blocks]

        with open(self.plots_nb_path, "w") as f:
            nbf.write(nb, f)

    def generate_code(self, item: pd.Series, plotting_lib: str) -> str:
        # Here we convert path to the data file to Unix-like path
        # this path is used then in jupyther notebook by pd.read_csv()
        # which accepts unix-like paths even on windows.
        # Unfortunately, windows-like paths cause an error.
        csv_path = self.csv_folder / f"data-{item['id']}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Unpacked csv datafile not found on {csv_path}")
        data_path_unix = str(PurePosixPath(csv_path))

        data_load_code = item["code_data"].replace("data.csv", data_path_unix)
        generated_code = item["code"]

        # Gather a code, adding index number at the first line as comment
        code_blocks = [f"# id = {item['id']}"]
        if "matplotlib" in plotting_lib or "seaborn" in plotting_lib:
            # We have to add "%matplotlib inline" line to the notebook cell
            # for major of antropic-claude generated code to be plotted
            code_blocks.append("%matplotlib inline")
        if "plotly" in plotting_lib:
            code_blocks.extend(
                ["import plotly.io as pio", 'pio.renderers.default = "png"']
            )
        # Main code
        code_blocks.extend([data_load_code, generated_code])
        # Resetting matplotlib global parameters
        if "matplotlib" in plotting_lib or "seaborn" in plotting_lib:
            code_blocks.append("plt.rcParams.update(plt.rcParamsDefault)")
        if "seaborn" in plotting_lib:
            code_blocks.append("sns.reset_orig()")
        # resetting all variables at the last line
        code_blocks.append("%reset -f")
        plot_code_nb = "\n".join(code_blocks)

        return plot_code_nb
    
    def build_debug_plots(self, dataset: pd.DataFrame) -> Path:
        """
        Generate and execute the plotting notebook for self_debug mode, output file starts with "debug_".
        """
        plotting_lib = self.config.plotting_lib.lower()
        setup_cell = []
        if "lets-plot" in plotting_lib:
            setup_cell = ["# Setup\n!pip install lets-plot\n"
                        "!jupyter nbextension install lets-plot --py --sys-prefix\n"
                        "!jupyter nbextension enable lets-plot --py --sys-prefix"]

        cells = setup_cell + dataset.apply(
            self.generate_code, axis=1, args=(plotting_lib,)
        ).tolist()
        self.build_new_nb(cells)

        debug_nb_path = self.plots_nb_path.with_name("debug_" + self.plots_nb_path.name)

        with open(self.plots_nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=10,
            interrupt_on_timeout=True,
            allow_errors=True,
            kernel_name="python3",
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": "."}})
        except CellExecutionError as e:
            print(f"[WARNING] Execution stopped early in debug: {e}")

        with open(debug_nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return debug_nb_path

    def build_plots(self, dataset: pd.DataFrame) -> Path:
        plotting_lib = self.config.plotting_lib.lower()
        setup_cell = []
        if "lets-plot" in plotting_lib:
            setup_cell = ["# Setup\n!pip install lets-plot\n"
                        "!jupyter nbextension install lets-plot --py --sys-prefix\n"
                        "!jupyter nbextension enable lets-plot --py --sys-prefix"]
        cells = setup_cell + dataset.apply(
            self.generate_code, axis=1, args=(plotting_lib,)
        ).tolist()
        self.build_new_nb(cells)

        with open(self.plots_nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=10,
            interrupt_on_timeout=True,
            allow_errors=True, 
            kernel_name="python3",
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": "."}})
        except CellExecutionError as e:
            print(f"[WARNING] Execution stopped early: {e}\nNotebook will still be saved.")

        with open(self.plots_nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        # ---------------------------------------

        return self.plots_nb_path

    @staticmethod
    def parse_plots_notebook(plots_nb_path: Path | None = None) -> pd.DataFrame:
        """
        Parses notebook with plotted plots and gathers the results to a json. Saves it.
        """

        with open(plots_nb_path) as f:
            nb = nbf.read(f, as_version=4)

        plot_results = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            # At the beginning of each cell I added "id = {index}".
            # if the cell does not begin with "# id = ", it does not correspond to datapoint we ignore it
            code = cell["source"].lstrip("\n")
            if not code.startswith("# id = "):
                continue
            # Extracting the index
            idx = int(code.split("\n")[0].lstrip("# id = "))
            cell_res = {"id": idx, "error": "", "plots_generated": []}

            images = []
            img_num = 0
            for output in cell["outputs"]:
                if output.output_type == "error":
                    traceback_raw = "\n".join(output.get("traceback", []))
                    cell_res["error"] = remove_ansi_escape(traceback_raw)
                    cell_res["error"] = remove_bytes_literal(cell_res["error"])
                elif (
                    output.output_type == "display_data" and "image/png" in output.data
                ):
                    image = output.data["image/png"]
                    images.append(image)
                    img_num += 1

            cell_res["plots_generated"] = images
            cell_res["has_plot"] = len(images) > 0
            plot_results.append(cell_res)

        return pd.DataFrame(plot_results)

    def draw_plots(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        model_name = dataset["model"].iloc[0].replace("/", "__")
        data_descriptor = dataset["data_descriptor"].iloc[0]
        plot_lib = self.config.plotting_lib.split(" ")[0]
        self.plots_nb_path, _ = add_index_to_filename(
            self.output_folder, f"plots_{data_descriptor}_{model_name}_{plot_lib}.ipynb"
        )
        self.build_plots(dataset)
        response = self.parse_plots_notebook(self.plots_nb_path)
        # Removing existing columns from dataset
        common_cols = dataset.columns.intersection(response.columns).drop("id")
        dataset = dataset.drop(columns=common_cols)
        dataset = dataset.merge(response, on="id", how="left")

        return dataset
    
    def draw_self_debug_plots(
        self,
        dataset: pd.DataFrame,
        attempt_id: int,
    ) -> pd.DataFrame:
        model_name = dataset["model"].iloc[0].replace("/", "__")
        data_descriptor = dataset["data_descriptor"].iloc[0]
        plot_lib = self.config.plotting_lib.split(" ")[0]
        os.makedirs(self.debug_plots_nb_path, exist_ok=True)
        self.plots_nb_path, _ = add_index_to_filename(
            self.debug_plots_nb_path, f"self_debug_attempt_{attempt_id}_{data_descriptor}_{model_name}_{plot_lib}.ipynb"
        )
        self.build_plots(dataset)
        response = self.parse_plots_notebook(self.plots_nb_path)
        # Removing existing columns from dataset
        common_cols = dataset.columns.intersection(response.columns).drop("id")
        dataset = dataset.drop(columns=common_cols)
        dataset = dataset.merge(response, on="id", how="left")

        return dataset
