import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from xgboost import XGBRFClassifier
except Exception:
    XGBRFClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import shap
except Exception:
    shap = None

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

TARGET_COL = "satisfaction"
DROP_COLS = ["Unnamed: 0", "id"]
# Use single-process CV for Streamlit+Windows stability (avoids loky temp-folder issues).
GRID_N_JOBS = 1
MODEL_ARTIFACT_PATH = Path("artifacts/model_results.joblib")


@st.cache_data
def load_data() -> pd.DataFrame:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combined = pd.concat([train_df, test_df], ignore_index=True)
    return combined


def split_feature_types(
    df: pd.DataFrame, target_col: str
) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }


def empty_metrics() -> dict[str, float]:
    return {
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC-ROC": np.nan,
    }


def roc_points(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, np.ndarray | float]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    return {"fpr": fpr, "tpr": tpr, "auc": auc_val}


@st.cache_resource
def load_saved_model_results(artifact_path: str) -> dict:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at {path}. "
            "Run offline training first to create it."
        )
    return joblib.load(path)


def train_models(
    df: pd.DataFrame, model_config: dict | None = None, _status_callback=None
) -> dict:
    def notify(message: str) -> None:
        if _status_callback is not None:
            _status_callback(message)

    default_config = {
        "cv_folds": 5,
        "tree_max_depth": [3, 5, 7, 10],
        "tree_min_samples_leaf": [5, 10, 20, 50],
        "rf_n_estimators": [50, 100, 200],
        "rf_max_depth": [3, 5, 8],
        "boost_n_estimators": [50, 100, 200],
        "boost_max_depth": [3, 4, 5, 6],
        "boost_learning_rate": [0.01, 0.05, 0.1],
        "mlp_hidden_units": 128,
        "mlp_epochs": 20,
        "mlp_batch_size": 256,
        "mlp_validation_split": 0.2,
        "mlp_patience": 3,
    }
    config = {**default_config, **(model_config or {})}

    notify("Preparing data split and preprocessing pipelines...")
    model_df = (
        df.drop(columns=DROP_COLS, errors="ignore").dropna(subset=[TARGET_COL]).copy()
    )

    y = (model_df[TARGET_COL].str.strip().str.lower() == "satisfied").astype(int)
    X = model_df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    lr_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    tree_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", lr_preprocess),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    notify("Running Logistic Regression baseline...")
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = classification_metrics(y_test, lr_pred, lr_proba)

    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", tree_preprocess),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]
    )
    tree_param_grid = {
        "clf__max_depth": config["tree_max_depth"],
        "clf__min_samples_leaf": config["tree_min_samples_leaf"],
    }
    tree_grid = GridSearchCV(
        estimator=tree_pipeline,
        param_grid=tree_param_grid,
        scoring="f1",
        cv=config["cv_folds"],
        n_jobs=GRID_N_JOBS,
        verbose=0,
    )
    notify("Running Decision Tree GridSearchCV (5-fold)...")
    tree_grid.fit(X_train, y_train)
    best_tree = tree_grid.best_estimator_
    tree_pred = best_tree.predict(X_test)
    tree_proba = best_tree.predict_proba(X_test)[:, 1]
    tree_metrics = classification_metrics(y_test, tree_pred, tree_proba)

    rf_compute_device = "CPU"
    rf_backend = "sklearn RandomForestClassifier"
    rf_used_fallback = False
    rf_param_grid = {
        "clf__n_estimators": config["rf_n_estimators"],
        "clf__max_depth": config["rf_max_depth"],
    }
    if XGBRFClassifier is not None:
        notify("Running Random Forest GridSearchCV on GPU...")
        try:
            rf_backend = "XGBoost Random Forest (XGBRFClassifier)"
            rf_pipeline = Pipeline(
                steps=[
                    ("preprocess", tree_preprocess),
                    (
                        "clf",
                        XGBRFClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            tree_method="hist",
                            device="cuda",
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            rf_grid = GridSearchCV(
                estimator=rf_pipeline,
                param_grid=rf_param_grid,
                scoring="f1",
                cv=config["cv_folds"],
                n_jobs=GRID_N_JOBS,
                verbose=0,
            )
            rf_grid.fit(X_train, y_train)
            rf_compute_device = "GPU"
        except Exception:
            notify("GPU not available for Random Forest. Falling back to CPU...")
            rf_used_fallback = True
            rf_pipeline = Pipeline(
                steps=[
                    ("preprocess", tree_preprocess),
                    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
                ]
            )
            rf_grid = GridSearchCV(
                estimator=rf_pipeline,
                param_grid=rf_param_grid,
                scoring="f1",
                cv=config["cv_folds"],
                n_jobs=GRID_N_JOBS,
                verbose=0,
            )
            rf_grid.fit(X_train, y_train)
            rf_compute_device = "CPU"
            rf_backend = "sklearn RandomForestClassifier"
    else:
        notify("XGBRFClassifier not available. Running sklearn Random Forest on CPU...")
        rf_pipeline = Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]
        )
        rf_grid = GridSearchCV(
            estimator=rf_pipeline,
            param_grid=rf_param_grid,
            scoring="f1",
            cv=config["cv_folds"],
            n_jobs=GRID_N_JOBS,
            verbose=0,
        )
        rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_
    if rf_backend.startswith("XGBoost"):
        try:
            best_rf.named_steps["clf"].set_params(device="cpu")
        except Exception:
            pass
    rf_pred = best_rf.predict(X_test)
    rf_proba = best_rf.predict_proba(X_test)[:, 1]
    rf_metrics = classification_metrics(y_test, rf_pred, rf_proba)

    boost_model_name = "Gradient Boosting (Fallback)"
    boost_compute_device = "CPU"
    boost_used_fallback = False
    boost_param_grid = {
        "clf__n_estimators": config["boost_n_estimators"],
        "clf__max_depth": config["boost_max_depth"],
        "clf__learning_rate": config["boost_learning_rate"],
    }

    def build_boost_grid(clf) -> GridSearchCV:
        boost_pipeline = Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                ("clf", clf),
            ]
        )
        return GridSearchCV(
            estimator=boost_pipeline,
            param_grid=boost_param_grid,
            scoring="f1",
            cv=config["cv_folds"],
            n_jobs=GRID_N_JOBS,
            verbose=0,
        )

    if XGBClassifier is not None:
        boost_model_name = "XGBoost"
        notify("Running XGBoost GridSearchCV on GPU...")
        try:
            xgb_gpu = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
                tree_method="hist",
                device="cuda",
            )
            boost_grid = build_boost_grid(xgb_gpu)
            boost_grid.fit(X_train, y_train)
            boost_compute_device = "GPU"
        except Exception:
            notify("GPU not available for XGBoost. Falling back to CPU...")
            xgb_cpu = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
                tree_method="hist",
                device="cpu",
            )
            boost_grid = build_boost_grid(xgb_cpu)
            boost_grid.fit(X_train, y_train)
            boost_compute_device = "CPU"
            boost_used_fallback = True
    elif LGBMClassifier is not None:
        boost_model_name = "LightGBM"
        notify("Running LightGBM GridSearchCV on GPU...")
        try:
            lgbm_gpu = LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                device="gpu",
            )
            boost_grid = build_boost_grid(lgbm_gpu)
            boost_grid.fit(X_train, y_train)
            boost_compute_device = "GPU"
        except Exception:
            notify("GPU not available for LightGBM. Falling back to CPU...")
            lgbm_cpu = LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                device="cpu",
            )
            boost_grid = build_boost_grid(lgbm_cpu)
            boost_grid.fit(X_train, y_train)
            boost_compute_device = "CPU"
            boost_used_fallback = True
    else:
        notify(
            "XGBoost/LightGBM not installed. Running Gradient Boosting fallback on CPU..."
        )
        boost_clf = GradientBoostingClassifier(random_state=42)
        boost_grid = build_boost_grid(boost_clf)
        boost_grid.fit(X_train, y_train)
        boost_used_fallback = True

    notify("Finalizing boosted-tree metrics...")
    best_boost = boost_grid.best_estimator_
    # If training used XGBoost on GPU, switch inference to CPU to avoid
    # device-mismatch warnings when the pipeline output is CPU memory.
    if boost_model_name == "XGBoost":
        try:
            best_boost.named_steps["clf"].set_params(device="cpu")
        except Exception:
            pass
    boost_pred = best_boost.predict(X_test)
    boost_proba = best_boost.predict_proba(X_test)[:, 1]
    boost_metrics = classification_metrics(y_test, boost_pred, boost_proba)

    mlp_available = tf is not None
    mlp_error = None
    mlp_metrics = empty_metrics()
    mlp_roc = None
    mlp_history = {}
    mlp_epochs = 0
    if mlp_available:
        notify("Running Neural Network (Keras MLP)...")
        try:
            mlp_preprocess = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_features,
                    ),
                    (
                        "cat",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                (
                                    "onehot",
                                    OneHotEncoder(
                                        handle_unknown="ignore", sparse_output=False
                                    ),
                                ),
                            ]
                        ),
                        cat_features,
                    ),
                ]
            )
            X_train_mlp = mlp_preprocess.fit_transform(X_train)
            X_test_mlp = mlp_preprocess.transform(X_test)
            if hasattr(X_train_mlp, "toarray"):
                X_train_mlp = X_train_mlp.toarray()
            if hasattr(X_test_mlp, "toarray"):
                X_test_mlp = X_test_mlp.toarray()

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)

            mlp_model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X_train_mlp.shape[1],)),
                    tf.keras.layers.Dense(
                        config["mlp_hidden_units"], activation="relu"
                    ),
                    tf.keras.layers.Dense(
                        config["mlp_hidden_units"], activation="relu"
                    ),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            mlp_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["mlp_patience"],
                restore_best_weights=True,
            )

            history = mlp_model.fit(
                X_train_mlp,
                y_train.to_numpy(),
                validation_split=config["mlp_validation_split"],
                epochs=config["mlp_epochs"],
                batch_size=config["mlp_batch_size"],
                verbose=0,
                callbacks=[early_stop],
            )
            mlp_history = history.history
            mlp_epochs = len(mlp_history.get("loss", []))

            mlp_proba = mlp_model.predict(X_test_mlp, verbose=0).ravel()
            mlp_pred = (mlp_proba >= 0.5).astype(int)
            mlp_metrics = classification_metrics(y_test, mlp_pred, mlp_proba)
            mlp_roc = roc_points(y_test, mlp_proba)
        except Exception as ex:
            mlp_error = str(ex)
    else:
        mlp_error = "TensorFlow is not installed in the current environment."

    return {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "num_features": num_features,
        "cat_features": cat_features,
        "y_test": y_test,
        "lr_metrics": lr_metrics,
        "lr_roc": roc_points(y_test, lr_proba),
        "lr_model": lr_pipeline,
        "tree_metrics": tree_metrics,
        "tree_roc": roc_points(y_test, tree_proba),
        "tree_best_params": tree_grid.best_params_,
        "best_tree": best_tree,
        "rf_metrics": rf_metrics,
        "rf_roc": roc_points(y_test, rf_proba),
        "rf_best_params": rf_grid.best_params_,
        "rf_compute_device": rf_compute_device,
        "rf_backend": rf_backend,
        "rf_used_fallback": rf_used_fallback,
        "best_rf": best_rf,
        "boost_model_name": boost_model_name,
        "boost_compute_device": boost_compute_device,
        "boost_used_fallback": boost_used_fallback,
        "boost_metrics": boost_metrics,
        "boost_roc": roc_points(y_test, boost_proba),
        "boost_best_params": boost_grid.best_params_,
        "best_boost": best_boost,
        "mlp_available": mlp_available,
        "mlp_error": mlp_error,
        "mlp_metrics": mlp_metrics,
        "mlp_roc": mlp_roc,
        "mlp_history": mlp_history,
        "mlp_epochs": mlp_epochs,
        "model_config": config,
    }


def main() -> None:
    st.title("Airline Passenger Satisfaction Dashboard")

    df = load_data()
    numeric_cols, categorical_cols = split_feature_types(df, TARGET_COL)
    if "model_results" not in st.session_state:
        try:
            st.session_state["model_results"] = load_saved_model_results(str(MODEL_ARTIFACT_PATH))
        except Exception:
            pass

    tabs = st.tabs(
        [
            "Executive Summary",
            "Descriptive Analytics",
            "Modeling (2.1-2.7)",
            "SHAP Analysis",
        ]
    )

    with tabs[0]:
        st.header("Executive Summary")
        st.subheader("Dataset and Prediction Task")
        st.markdown(
            """
            This project analyzes an airline passenger satisfaction dataset built from post-flight survey responses.
            Each row represents one passenger experience and includes demographics (`Gender`, `Age`), trip context (`Type of Travel`, `Class`, `Flight Distance`),
            service-quality ratings (such as `Online boarding`, `Seat comfort`, `Inflight wifi service`, `Cleanliness`), and operational reliability measures
            (`Departure Delay in Minutes`, `Arrival Delay in Minutes`). The prediction target is `satisfaction`, with two classes:
            `satisfied` and `neutral or dissatisfied`. The core business question is whether we can predict satisfaction reliably and identify which factors most strongly
            move a passenger toward a positive or negative experience.
            """
        )
        st.subheader("Why This Problem Matters (The So What)")
        st.markdown(
            """
            This problem matters because customer satisfaction directly influences repeat purchase behavior, loyalty program value, and brand perception in a highly competitive airline market.
            If airlines can predict likely dissatisfaction before or during the journey, they can intervene earlier with service recovery actions, staffing adjustments, digital experience fixes,
            or operational improvements. In practice, this supports better allocation of limited improvement budgets toward the touchpoints with the highest impact on customer outcomes. Meanwhile, using SHAP in this dashboard will help customers or users to get a good idea of what they will experience for their next flight.
            """
        )
        st.subheader("Approach and Key Findings")
        st.markdown(
            """
            The analysis combines descriptive analytics and supervised machine learning. We first profile the dataset structure, target balance, feature distributions,
            and correlations to understand major behavioral and operational patterns. We then train and compare multiple models (Logistic Regression, Decision Tree, Random Forest,
            Boosted Trees, and MLP), using cross-validation-based tuning and held-out test evaluation with classification metrics including F1 and AUC-ROC.
            """
        )
        if "model_results" in st.session_state:
            metrics = {
                "Logistic Regression": st.session_state["model_results"]["lr_metrics"][
                    "F1"
                ],
                "Decision Tree": st.session_state["model_results"]["tree_metrics"][
                    "F1"
                ],
                "Random Forest": st.session_state["model_results"]["rf_metrics"]["F1"],
                st.session_state["model_results"]["boost_model_name"]: st.session_state[
                    "model_results"
                ]["boost_metrics"]["F1"],
                "MLP": st.session_state["model_results"]["mlp_metrics"]["F1"],
            }
            best_name = max(metrics, key=metrics.get)
            st.markdown(
                f"""
                Based on the current run, **{best_name}** is the top performer by F1 score, showing that non-linear models capture customer-satisfaction drivers better than a purely linear baseline.
                The trade-off is interpretability versus predictive power: linear and shallow tree models are easier to explain, while boosted and neural models generally improve accuracy at the cost of
                greater complexity and training overhead. The SHAP tab complements this by translating model behavior into feature-level explanations that decision-makers can act on.
                """
            )
        else:
            st.markdown(
                """
                Initial results from this workflow typically show stronger performance from tree-ensemble and boosted approaches than from a linear baseline,
                indicating meaningful non-linear interactions between service ratings, travel context, and delay features. The app includes model comparison and SHAP analysis to balance performance
                with transparency, so stakeholders can see both what predicts satisfaction and why.
                """
            )
        st.subheader("Data Source")
        st.markdown(
            """
            The datasheet comes from Kaggle.
            https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data?select=train.csv
            """
        )

    with tabs[1]:
        st.header("Dataset Introduction")
        st.markdown(
            """
            This dataset contains airline passenger satisfaction survey responses, merged from `train.csv` and `test.csv`.
            Each row is one passenger trip with demographic information, trip context, service ratings, delays, and the final satisfaction label.

            The prediction target (dependent variable) is **`satisfaction`**, with classes:
            - `satisfied`
            - `neutral or dissatisfied`

            This prediction task is impactful because it helps airlines identify the strongest drivers of customer experience,
            prioritize service improvements, and target interventions that can improve retention and loyalty.
            """
        )

        total_rows = len(df)
        total_features = df.shape[1] - 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Features (excluding target)", total_features)
        c4.metric("Target", TARGET_COL)

        st.markdown(
            f"**Feature Type Breakdown:** {len(numeric_cols)} numerical, {len(categorical_cols)} categorical "
            f"(excluding target)."
        )

        with st.expander("View column details"):
            st.write("Numerical features", numeric_cols)
            st.write("Categorical features", categorical_cols)

        st.divider()

        st.header("Target Distribution")
        target_counts = df[TARGET_COL].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=df,
            x=TARGET_COL,
            hue=TARGET_COL,
            order=target_counts.index,
            palette="Set2",
            legend=False,
            ax=ax,
        )
        ax.set_title("Satisfaction Class Distribution")
        ax.set_xlabel("Satisfaction")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=10)
        st.pyplot(fig)

        majority_pct = (target_counts.max() / target_counts.sum()) * 100
        st.markdown(
            f"The classes are not perfectly balanced, with the majority class representing about **{majority_pct:.1f}%** "
            "of observations. This is a moderate imbalance rather than an extreme one. "
            "For modeling, stratified train/test splits and class-aware metrics (F1, ROC-AUC, recall) should be used; "
            "if needed, class weights or resampling can be added."
        )

        st.divider()

        st.header("Feature Distributions and Relationships")

        viz1_col, viz2_col = st.columns(2)
        with viz1_col:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=df,
                x=TARGET_COL,
                y="Flight Distance",
                hue=TARGET_COL,
                palette="pastel",
                legend=False,
                ax=ax1,
            )
            ax1.set_title("Flight Distance by Satisfaction")
            ax1.tick_params(axis="x", rotation=10)
            st.pyplot(fig1)
            st.markdown(
                "Passengers who are satisfied tend to show a higher median flight distance, with a wider spread in long-haul travel. "
                "This suggests trip profile may affect expectation levels and service exposure. "
                "Distance alone is not determinative, but it likely interacts with class and service ratings."
            )

        with viz2_col:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.violinplot(
                data=df,
                x="Class",
                y="Online boarding",
                hue=TARGET_COL,
                split=True,
                inner="quart",
                ax=ax2,
            )
            ax2.set_title("Online Boarding Ratings by Class and Satisfaction")
            ax2.set_ylim(-0.2, 5.2)
            st.pyplot(fig2)
            st.markdown(
                "Online boarding ratings separate satisfied and dissatisfied passengers clearly across all cabin classes. "
                "Satisfied passengers cluster around higher scores, while dissatisfied passengers concentrate at lower-to-mid ratings. "
                "This pattern indicates digital and boarding experience is a strong candidate predictor."
            )

        viz3_col, viz4_col = st.columns(2)
        with viz3_col:
            sat_rate_by_travel = (
                df.groupby("Type of Travel")[TARGET_COL]
                .value_counts(normalize=True)
                .rename("rate")
                .reset_index()
            )
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=sat_rate_by_travel,
                x="Type of Travel",
                y="rate",
                hue=TARGET_COL,
                palette="Set2",
                ax=ax3,
            )
            ax3.set_title("Satisfaction Rate by Type of Travel")
            ax3.set_ylabel("Proportion")
            ax3.set_xlabel("Type of Travel")
            ax3.tick_params(axis="x", rotation=10)
            st.pyplot(fig3)
            st.markdown(
                "Business travel shows a meaningfully higher satisfaction share than personal travel. "
                "The gap suggests purpose-of-travel is associated with different service expectations and trip conditions. "
                "This variable is likely valuable in both feature engineering and segment-level analysis."
            )

        with viz4_col:
            sampled = df.sample(n=min(15000, len(df)), random_state=42)
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=sampled,
                x="Age",
                y="Flight Distance",
                hue=TARGET_COL,
                alpha=0.45,
                s=20,
                ax=ax4,
            )
            ax4.set_title("Age vs Flight Distance by Satisfaction (Sampled)")
            st.pyplot(fig4)
            st.markdown(
                "The age-distance landscape is mixed, but satisfied passengers appear more frequently in medium-to-long distance regions. "
                "No single linear boundary separates classes, which supports using non-linear models or interaction features. "
                "The plot also shows substantial overlap, indicating that service-quality variables are likely more predictive than demographics alone."
            )

        st.divider()

        st.header("Correlation Heatmap")
        numeric_for_corr = df.select_dtypes(include=[np.number]).drop(
            columns=DROP_COLS, errors="ignore"
        )
        corr = numeric_for_corr.corr(numeric_only=True)

        fig5, ax5 = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax5)
        ax5.set_title("Correlation Matrix (Numerical Features)")
        st.pyplot(fig5)

        abs_corr = corr.abs().where(~np.eye(corr.shape[0], dtype=bool))
        top_pairs = (
            abs_corr.stack().sort_values(ascending=False).drop_duplicates().head(5)
        )
        st.markdown("**Strongest observed correlations (absolute):**")
        for (f1, f2), val in top_pairs.items():
            st.markdown(f"- `{f1}` and `{f2}`: {val:.2f}")

        st.markdown(
            "Higher correlations often appear among related service-rating variables, suggesting latent dimensions of overall service quality. "
            "Delay variables also tend to correlate with each other, which is operationally expected. "
            "For modeling, this indicates possible multicollinearity in linear models and suggests benefit from regularization or tree-based methods."
        )

    with tabs[2]:
        st.header("2.1 Data Preparation")
        st.markdown(
            """
            **Target (`y`)**: `satisfaction` converted to binary (`satisfied`=1, `neutral or dissatisfied`=0).  
            **Features (`X`)**: all columns except `satisfaction`, with `Unnamed: 0` and `id` removed.

            **Split**: 70/30 train/test using `random_state=42` and stratification on the target.  
            **Preprocessing**:
            - Numerical: median imputation (and standard scaling for Logistic Regression).
            - Categorical: most-frequent imputation + one-hot encoding.

            This handles missing values safely, preserves categorical information, and ensures the linear baseline model is properly scaled.
            """
        )

        st.subheader("Pre-Trained Model Artifacts")
        st.markdown(
            "This app only loads saved models and metrics from disk. It does not retrain models during Streamlit runtime."
        )
        st.code(f"Artifact path: {MODEL_ARTIFACT_PATH}")
        if st.button("Reload saved artifacts"):
            st.cache_resource.clear()
            if "model_results" in st.session_state:
                del st.session_state["model_results"]
            st.rerun()

        try:
            model_results = load_saved_model_results(str(MODEL_ARTIFACT_PATH))
            st.session_state["model_results"] = model_results
            st.success("Saved model artifacts loaded successfully.")
        except Exception as ex:
            st.error(
                f"Could not load saved artifacts: {ex}\n\n"
                "Run offline training first using `python train_and_save_models.py`, then reload this tab."
            )
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Rows", f"{model_results['X_train_shape'][0]:,}")
        c2.metric("Test Rows", f"{model_results['X_test_shape'][0]:,}")
        c3.metric("Numerical Features", len(model_results["num_features"]))
        c4.metric("Categorical Features", len(model_results["cat_features"]))
        st.caption(f"Current tuning config: `{model_results.get('model_config', {})}`")

        st.divider()
        st.header("2.2-2.6 Model Results and Comparison")
        st.markdown(
            "The table below compares pre-trained models loaded from disk. "
            "Use the selector to inspect one model at a time (default is Logistic Regression Baseline)."
        )

        model_metrics = {
            "Logistic Regression Baseline": model_results["lr_metrics"],
            "Decision Tree / CART": model_results["tree_metrics"],
            "Random Forest": model_results["rf_metrics"],
            model_results["boost_model_name"]: model_results["boost_metrics"],
            "Neural Network (MLP)": model_results["mlp_metrics"],
        }
        comparison_df = pd.DataFrame(model_metrics).T
        st.dataframe(comparison_df.style.format("{:.4f}"), width="stretch")

        st.divider()
        st.subheader("2.7 Model Comparison Summary")
        summary_df = comparison_df.copy()
        st.dataframe(summary_df.style.format("{:.4f}"), width="stretch")

        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
        f1_plot_df = summary_df.reset_index().rename(columns={"index": "Model"})
        sns.barplot(data=f1_plot_df, x="Model", y="F1", palette="Blues_d", ax=ax_cmp)
        ax_cmp.set_title("F1 Score Comparison Across Models")
        ax_cmp.set_xlabel("Model")
        ax_cmp.set_ylabel("F1")
        ax_cmp.tick_params(axis="x", rotation=20)
        st.pyplot(fig_cmp)

        valid_f1 = summary_df["F1"].dropna()
        if len(valid_f1) > 0:
            best_model = valid_f1.idxmax()
            best_f1 = valid_f1.max()
            baseline_f1 = summary_df.loc["Logistic Regression Baseline", "F1"]
            uplift = best_f1 - baseline_f1
            st.markdown(
                f"Best overall performance is from **{best_model}** with **F1 = {best_f1:.4f}**. "
                f"Compared with the Logistic Regression baseline (F1 = {baseline_f1:.4f}), "
                f"the uplift is **{uplift:.4f}**. "
                "Tree/boosted models typically improve predictive power but are less interpretable than linear models, "
                "while MLPs can be strong performers but add training complexity and lower explainability."
            )

        st.markdown("### Model Detail Viewer")
        st.info(
            "Choose one model below to display its detailed metrics, plots, and interpretation."
        )
        selected_model = st.selectbox(
            "Select a model to show details",
            options=[
                "Logistic Regression Baseline",
                "Decision Tree / CART",
                "Random Forest",
                model_results["boost_model_name"],
                "Neural Network (MLP)",
            ],
            index=0,
        )
        st.success(f"Currently showing details for: **{selected_model}**")

        if selected_model == "Logistic Regression Baseline":
            st.subheader("2.2 Logistic Regression Baseline")
            st.write(
                "Best hyperparameters: default Logistic Regression settings (`max_iter=2000`)."
            )
            st.dataframe(
                pd.DataFrame(
                    [model_results["lr_metrics"]], index=[selected_model]
                ).style.format("{:.4f}"),
                width="stretch",
            )

        elif selected_model == "Decision Tree / CART":
            st.subheader("2.3 Decision Tree / CART (5-Fold GridSearchCV)")
            st.markdown(
                "Grid searched over `max_depth` = [3, 5, 7, 10] and `min_samples_leaf` = [5, 10, 20, 50] "
                "with `scoring='f1'` and `cv=5`."
            )
            st.write("Best hyperparameters:", model_results["tree_best_params"])
            st.dataframe(
                pd.DataFrame(
                    [model_results["tree_metrics"]], index=[selected_model]
                ).style.format("{:.4f}"),
                width="stretch",
            )

            st.subheader("Best Tree Visualization")
            best_tree_pipeline = model_results["best_tree"]
            tree_model = best_tree_pipeline.named_steps["clf"]
            tree_preprocess = best_tree_pipeline.named_steps["preprocess"]
            feature_names = tree_preprocess.get_feature_names_out()
            tree_depth = tree_model.get_depth()

            fig_tree, ax_tree = plt.subplots(figsize=(24, 12))
            plot_tree(
                tree_model,
                feature_names=feature_names,
                class_names=["neutral or dissatisfied", "satisfied"],
                filled=True,
                rounded=True,
                fontsize=7,
                max_depth=None if tree_depth <= 5 else 3,
                ax=ax_tree,
            )
            ax_tree.set_title(
                "Best Decision Tree"
                if tree_depth <= 5
                else "Best Decision Tree (Truncated to First 4 Levels for Readability)"
            )
            st.pyplot(fig_tree)

        elif selected_model == "Random Forest":
            st.subheader("2.4 Random Forest (5-Fold GridSearchCV)")
            st.markdown(
                "Grid searched over `n_estimators` = [50, 100, 200] and `max_depth` = [3, 5, 8] "
                "with `scoring='f1'` and `cv=5`."
            )
            st.caption(f"Backend: **{model_results['rf_backend']}**")
            st.caption(
                f"Compute target for Random Forest: **{model_results['rf_compute_device']}**"
            )
            if model_results["rf_used_fallback"]:
                st.info(
                    "GPU execution was attempted first, then automatically switched to CPU."
                )
            st.write("Best hyperparameters:", model_results["rf_best_params"])
            st.dataframe(
                pd.DataFrame(
                    [model_results["rf_metrics"]], index=[selected_model]
                ).style.format("{:.4f}"),
                width="stretch",
            )

            roc_data = model_results["rf_roc"]
            fig_rf_roc, ax_rf_roc = plt.subplots(figsize=(7, 5))
            ax_rf_roc.plot(
                roc_data["fpr"], roc_data["tpr"], label=f"AUC = {roc_data['auc']:.4f}"
            )
            ax_rf_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_rf_roc.set_title("Random Forest ROC Curve")
            ax_rf_roc.set_xlabel("False Positive Rate")
            ax_rf_roc.set_ylabel("True Positive Rate")
            ax_rf_roc.legend(loc="lower right")
            st.pyplot(fig_rf_roc)

        elif selected_model == model_results["boost_model_name"]:
            st.subheader(
                f"2.5 Boosted Trees - {model_results['boost_model_name']} (5-Fold GridSearchCV)"
            )
            st.markdown(
                "Grid searched over `n_estimators` = [50, 100, 200], `max_depth` = [3, 4, 5, 6], "
                "and `learning_rate` = [0.01, 0.05, 0.1] with `scoring='f1'` and `cv=5`."
            )
            if model_results["boost_model_name"] == "Gradient Boosting (Fallback)":
                st.warning(
                    "XGBoost/LightGBM was not available in this environment, so a Gradient Boosting fallback was used."
                )
            st.caption(
                f"Compute target for boosted model: **{model_results['boost_compute_device']}**"
            )
            if (
                model_results["boost_used_fallback"]
                and model_results["boost_model_name"] != "Gradient Boosting (Fallback)"
            ):
                st.info(
                    "GPU execution was attempted first, then automatically switched to CPU."
                )
            st.write("Best hyperparameters:", model_results["boost_best_params"])
            st.dataframe(
                pd.DataFrame(
                    [model_results["boost_metrics"]], index=[selected_model]
                ).style.format("{:.4f}"),
                width="stretch",
            )

            roc_data = model_results["boost_roc"]
            fig_boost_roc, ax_boost_roc = plt.subplots(figsize=(7, 5))
            ax_boost_roc.plot(
                roc_data["fpr"], roc_data["tpr"], label=f"AUC = {roc_data['auc']:.4f}"
            )
            ax_boost_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_boost_roc.set_title(f"{model_results['boost_model_name']} ROC Curve")
            ax_boost_roc.set_xlabel("False Positive Rate")
            ax_boost_roc.set_ylabel("True Positive Rate")
            ax_boost_roc.legend(loc="lower right")
            st.pyplot(fig_boost_roc)

        else:
            st.subheader("2.6 Neural Network - MLP (Keras)")
            st.markdown(
                "Architecture: input layer, **Dense(128, ReLU)**, **Dense(128, ReLU)**, "
                "and **Dense(1, sigmoid)** output. Loss is `binary_crossentropy` with `Adam` optimizer."
            )
            if (
                not model_results["mlp_available"]
                or model_results["mlp_error"] is not None
            ):
                st.warning(f"MLP model not available: {model_results['mlp_error']}")
            else:
                st.write(f"Training epochs completed: {model_results['mlp_epochs']}")
                st.dataframe(
                    pd.DataFrame(
                        [model_results["mlp_metrics"]], index=[selected_model]
                    ).style.format("{:.4f}"),
                    width="stretch",
                )

                history = model_results["mlp_history"]
                fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))
                ax_hist[0].plot(history.get("loss", []), label="Train Loss")
                ax_hist[0].plot(history.get("val_loss", []), label="Val Loss")
                ax_hist[0].set_title("MLP Loss Curve")
                ax_hist[0].set_xlabel("Epoch")
                ax_hist[0].set_ylabel("Binary Cross-Entropy")
                ax_hist[0].legend()

                ax_hist[1].plot(history.get("accuracy", []), label="Train Accuracy")
                ax_hist[1].plot(history.get("val_accuracy", []), label="Val Accuracy")
                ax_hist[1].set_title("MLP Accuracy Curve")
                ax_hist[1].set_xlabel("Epoch")
                ax_hist[1].set_ylabel("Accuracy")
                ax_hist[1].legend()
                st.pyplot(fig_hist)

                if model_results["mlp_roc"] is not None:
                    roc_data = model_results["mlp_roc"]
                    fig_mlp_roc, ax_mlp_roc = plt.subplots(figsize=(7, 5))
                    ax_mlp_roc.plot(
                        roc_data["fpr"],
                        roc_data["tpr"],
                        label=f"AUC = {roc_data['auc']:.4f}",
                    )
                    ax_mlp_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    ax_mlp_roc.set_title("MLP ROC Curve")
                    ax_mlp_roc.set_xlabel("False Positive Rate")
                    ax_mlp_roc.set_ylabel("True Positive Rate")
                    ax_mlp_roc.legend(loc="lower right")
                    st.pyplot(fig_mlp_roc)

    with tabs[3]:
        st.header("SHAP Analysis")
        st.markdown(
            "SHAP overview uses XGBoost. Interactive prediction lets you choose a model and custom inputs."
        )

        if "model_results" not in st.session_state:
            st.warning(
                "Run models in the Modeling tab first, then return here for SHAP analysis."
            )
            return
        if shap is None:
            st.error(
                "SHAP is not installed. Install it with: `python -m pip install shap`"
            )
            return

        model_results = st.session_state["model_results"]
        model_df = (
            df.drop(columns=DROP_COLS, errors="ignore")
            .dropna(subset=[TARGET_COL])
            .copy()
        )
        y = (model_df[TARGET_COL].str.strip().str.lower() == "satisfied").astype(int)
        X = model_df.drop(columns=[TARGET_COL])
        _, X_test, _, _ = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )

        st.subheader("Global SHAP (XGBoost)")
        if model_results.get("boost_model_name") != "XGBoost":
            st.warning(
                "XGBoost is not the current boosted model, so global SHAP summary/bar plots are unavailable."
            )
        else:
            shap_sample_size = st.slider(
                "XGBoost SHAP sample size (test rows)",
                min_value=200,
                max_value=3000,
                value=1000,
                step=100,
            )
            if st.button("Generate XGBoost SHAP Summary/Bar", type="primary"):
                boost_pipeline = model_results["best_boost"]
                boost_preprocess = boost_pipeline.named_steps["preprocess"]
                boost_model = boost_pipeline.named_steps["clf"]
                X_test_enc = boost_preprocess.transform(X_test)
                feature_names = boost_preprocess.get_feature_names_out()
                sample_n = min(shap_sample_size, X_test_enc.shape[0])
                sample_idx = np.random.RandomState(42).choice(
                    X_test_enc.shape[0], size=sample_n, replace=False
                )
                X_shap = X_test_enc[sample_idx]
                if hasattr(X_shap, "toarray"):
                    X_shap = X_shap.toarray()
                X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

                with st.status(
                    "Computing global SHAP values...", expanded=True
                ) as shap_status:
                    try:
                        explainer = shap.Explainer(boost_model, X_shap_df)
                        shap_values = explainer(X_shap_df)
                        shap_status.update(
                            label="Global SHAP complete.", state="complete"
                        )
                    except Exception as ex:
                        shap_status.update(label="Global SHAP failed.", state="error")
                        st.error(f"SHAP failed: {ex}")
                        return

                fig_swarm = plt.figure(figsize=(11, 6))
                shap.plots.beeswarm(shap_values, max_display=20, show=False)
                st.pyplot(fig_swarm)

                fig_bar = plt.figure(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=20, show=False)
                st.pyplot(fig_bar)

        st.divider()
        st.subheader("Interactive Prediction + Custom SHAP Waterfall")
        model_choice = st.selectbox(
            "Choose model for prediction",
            options=[
                "Logistic Regression Baseline",
                "Decision Tree / CART",
                "Random Forest",
                model_results["boost_model_name"],
            ],
            index=0,
        )

        feature_defaults = X.median(numeric_only=True).to_dict()
        for col in X.select_dtypes(exclude=[np.number]).columns:
            mode_series = X[col].mode(dropna=True)
            feature_defaults[col] = (
                mode_series.iloc[0]
                if not mode_series.empty
                else X[col].dropna().iloc[0]
            )

        c1, c2 = st.columns(2)
        with c1:
            age_val = st.slider(
                "Age",
                int(X["Age"].min()),
                int(X["Age"].max()),
                int(feature_defaults.get("Age", 40)),
            )
            dist_val = st.slider(
                "Flight Distance",
                int(X["Flight Distance"].min()),
                int(X["Flight Distance"].max()),
                int(feature_defaults.get("Flight Distance", 1000)),
            )
            dep_delay_val = st.slider(
                "Departure Delay in Minutes",
                int(X["Departure Delay in Minutes"].min()),
                int(X["Departure Delay in Minutes"].max()),
                int(feature_defaults.get("Departure Delay in Minutes", 0)),
            )
            arr_delay_val = st.slider(
                "Arrival Delay in Minutes",
                int(X["Arrival Delay in Minutes"].fillna(0).min()),
                int(X["Arrival Delay in Minutes"].fillna(0).max()),
                int(feature_defaults.get("Arrival Delay in Minutes", 0)),
            )
            online_boarding_val = st.slider(
                "Online boarding", 0, 5, int(feature_defaults.get("Online boarding", 3))
            )
            seat_comfort_val = st.slider(
                "Seat comfort", 0, 5, int(feature_defaults.get("Seat comfort", 3))
            )

        with c2:
            class_val = st.selectbox(
                "Class", options=sorted(X["Class"].dropna().unique().tolist()), index=0
            )
            travel_type_val = st.selectbox(
                "Type of Travel",
                options=sorted(X["Type of Travel"].dropna().unique().tolist()),
                index=0,
            )
            customer_type_val = st.selectbox(
                "Customer Type",
                options=sorted(X["Customer Type"].dropna().unique().tolist()),
                index=0,
            )
            gender_val = st.selectbox(
                "Gender",
                options=sorted(X["Gender"].dropna().unique().tolist()),
                index=0,
            )
            wifi_val = st.slider(
                "Inflight wifi service",
                0,
                5,
                int(feature_defaults.get("Inflight wifi service", 3)),
            )
            cleanliness_val = st.slider(
                "Cleanliness", 0, 5, int(feature_defaults.get("Cleanliness", 3))
            )

        custom_row = feature_defaults.copy()
        custom_row.update(
            {
                "Age": age_val,
                "Flight Distance": dist_val,
                "Departure Delay in Minutes": dep_delay_val,
                "Arrival Delay in Minutes": arr_delay_val,
                "Online boarding": online_boarding_val,
                "Seat comfort": seat_comfort_val,
                "Class": class_val,
                "Type of Travel": travel_type_val,
                "Customer Type": customer_type_val,
                "Gender": gender_val,
                "Inflight wifi service": wifi_val,
                "Cleanliness": cleanliness_val,
            }
        )
        custom_df = pd.DataFrame([custom_row], columns=X.columns)

        model_map = {
            "Logistic Regression Baseline": model_results["lr_model"],
            "Decision Tree / CART": model_results["best_tree"],
            "Random Forest": model_results["best_rf"],
            model_results["boost_model_name"]: model_results["best_boost"],
        }
        selected_pipeline = model_map[model_choice]
        pred_proba = float(selected_pipeline.predict_proba(custom_df)[0, 1])
        pred_class = "satisfied" if pred_proba >= 0.5 else "neutral or dissatisfied"
        st.markdown(f"**Predicted class:** `{pred_class}`")
        st.markdown(f"**Predicted probability (satisfied):** `{pred_proba:.4f}`")

        if model_choice in [
            "Decision Tree / CART",
            "Random Forest",
            model_results["boost_model_name"],
        ]:
            try:
                pre = selected_pipeline.named_steps["preprocess"]
                clf = selected_pipeline.named_steps["clf"]
                X_bg = pre.transform(
                    X_test.sample(n=min(300, len(X_test)), random_state=42)
                )
                X_user = pre.transform(custom_df)
                feature_names = pre.get_feature_names_out()
                if hasattr(X_bg, "toarray"):
                    X_bg = X_bg.toarray()
                if hasattr(X_user, "toarray"):
                    X_user = X_user.toarray()
                X_bg_df = pd.DataFrame(X_bg, columns=feature_names)
                X_user_df = pd.DataFrame(X_user, columns=feature_names)

                explainer = shap.Explainer(clf, X_bg_df)
                user_sv = explainer(X_user_df)
                st.subheader("SHAP Waterfall for Custom Input")
                fig_user_wf = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(user_sv[0], max_display=15, show=False)
                st.pyplot(fig_user_wf)
            except Exception as ex:
                st.warning(
                    f"Could not generate custom SHAP waterfall for {model_choice}: {ex}"
                )
        else:
            st.info(
                "Custom SHAP waterfall is enabled for tree-based models in this app."
            )


if __name__ == "__main__":
    main()
