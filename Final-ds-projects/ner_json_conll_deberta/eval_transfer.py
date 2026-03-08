import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nerjson_conll.evaluation.eval_transfer_coarse import main
if __name__ == "__main__":
    main()
