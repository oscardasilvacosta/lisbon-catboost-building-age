# Lisbon CatBoost building-age model

Training code and the fitted construction-age classifier for the Lisbon building-scale seismic vulnerability study.

- `train_catboost.py` runs the BGRI-grouped model selection, held-out evaluation, and final refit on all labelled records.
- `best_model_config.json` records the selected predictor order and hyperparameters.
- [`catboost_age_model_FULL_100pct.cbm`](https://github.com/oscardasilvacosta/lisbon-catboost-building-age/releases/download/v1.0.0/catboost_age_model_FULL_100pct.cbm) is the final model refitted on all 19,738 labelled buildings. Download it from the release and place it alongside the configuration file before running inference.

The script takes a labelled CSV containing the predictors, `Age_Buildings`, and `BGRI21` for grouping. Supply the training inventory separately.

```sh
python -m pip install -r requirements.txt
python train_catboost.py path/to/labelled_buildings.csv
```

For inference, load the fitted model with CatBoost and supply predictor columns in the order listed in the configuration. `Type` is categorical. The prediction codes 0 through 6 correspond to pre-18th century, 18th century, 19th century, 1900–1930, 1930–1960, 1960–1983, and post-1983.

```python
import json
from catboost import CatBoostClassifier

with open("best_model_config.json", encoding="utf-8") as file:
    features = json.load(file)["features"]

model = CatBoostClassifier()
model.load_model("catboost_age_model_FULL_100pct.cbm")
# buildings is a pandas DataFrame containing the prepared predictors.
# codes = model.predict(buildings[features]).ravel()
```

Model SHA-256

```text
76ff74cb1cba7450f887a66bce60ce3dd22e0bb8dd37058054147299060d0daf
```
