from absl import app, flags

from src.network.testing_single import ExcuteSingleTestingProcess
from src.utils.result_visualisation import ResultCorrectVisualsation
from src.utils.auto_label import AutoLabel

# -------------
FLAGS = flags.FLAGS

# -------------
def main(_):
    ExcuteSingleTestingProcess()
    ResultCorrectVisualsation()

# -------------
if __name__ == "__main__":
    app.run(main)
